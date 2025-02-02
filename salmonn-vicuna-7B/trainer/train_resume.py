# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

from utils import *
from config import Config
from dist_utils import get_rank, init_distributed_mode
from models import load_model
from dataset import SALMONNDataset
from runner_resume import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--dryrun", action='store_true', help='if True, use dummy model and skip forward/backward')
    
    # 명시적으로 resume_checkpoint 인자 추가
    parser.add_argument("--resume_checkpoint", type=str, default=None, help='path to checkpoint to resume training')

    # torchrun의 추가 인자 처리
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--master_port", type=str, default=None)

    # 디버깅용 print 추가
    args, unknown = parser.parse_known_args()
    print("Parsed arguments:", args)
    print("Unknown arguments:", unknown)

    return args


def setup_seeds(config):
    seed = config.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main():
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    # load config
    args = parse_args()
    cfg = Config(args)
    run_config = cfg.config.run
    model_config = cfg.config.model
    data_config = cfg.config.datasets

    # 체크포인트 경로를 config에 추가
    if args.resume_checkpoint:
        model_config['ckpt'] = args.resume_checkpoint

    # initialize distributed training
    init_distributed_mode(run_config)
    setup_seeds(run_config)
    setup_logger() # set after init_distributed_mode() to only log on master.

    # Wandb logger
    global_rank = int(os.environ["RANK"])
    print(global_rank)
    if global_rank == 0:
        wandb.login(key="6672bc576d4aaaa87a27ab89d9229eb2c69ac0a4")
        wandb.init(project="audio_lm", name=run_config.exp_name)

    # print config
    cfg.pretty_print()

    # build datasets
    datasets = {
        "train": SALMONNDataset(data_config.prefix, data_config.train_ann_path, data_config.whisper_path),
        "valid": SALMONNDataset(data_config.prefix, data_config.valid_ann_path, data_config.whisper_path)
    }

    # build runner
    if not args.dryrun:
        # from_config()에서 반환된 튜플에서 모델과 시작 에폭 추출
        model, start_epoch = load_model(model_config)
        runner = Runner(cfg, model, datasets, job_id, args.dryrun, start_epoch=start_epoch)
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("apple/OpenELM-270M-Instruct", trust_remote_code=True)
        runner = Runner(cfg, model, datasets, job_id, args.dryrun)

    # train
    runner.train()


if __name__ == "__main__":
    main()