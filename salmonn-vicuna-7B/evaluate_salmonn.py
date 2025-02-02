import argparse
import json
import random
import sys
import torch
import torch.distributed as dist
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# # Add custom module path
# sys.path.append(str(Path(__file__).parent / "audiolm-trainer"))

# Custom modules
from salmonn_utils import SALMONNTestDataset, load_preprocessor, load_model
from config import Config
from utils import get_dataloader, prepare_sample
from metrics import compute_wer, compute_spider


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg-path", 
        type=str, 
        help='path to configuration file', 
        default='salmonn_eval_config.yaml'
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument("--task", type=str, default=None, help="Task to evaluate", choices=['asr', 'aac'])
    parser.add_argument("--skip_scoring", action='store_true', help='if True, skip scoring after inference')
    return parser.parse_args()


def get_dataset(dataset_cfg, run_cfg, task):
    testset = SALMONNTestDataset(
        dataset_cfg.prefix, dataset_cfg.test_ann_path, dataset_cfg.whisper_path, task
    )

    test_loader = get_dataloader(testset, run_cfg, is_train=False, use_distributed=False)
    return test_loader


def main(args):
    # 분산 처리 초기화 추가
    if torch.cuda.device_count() > 1:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=int(os.environ.get('WORLD_SIZE', 1)),
            rank=local_rank
        )
    
    cfg = Config(args)
    
    # Load models
    salmonn_preprocessor = load_preprocessor(cfg)
    llama_model, tokenizer = load_model(salmonn_preprocessor)
    salmonn_preprocessor.llama_model = llama_model

    # Load data
    dataloader = get_dataset(cfg.config.datasets, cfg.config.run, args.task)

    # Load prompt from config
    default_prompt_path = Path(__file__).parent / 'audiolm-trainer' / 'prompts' / 'test_prompt.json'
    prompt_path = Path(cfg.config.model.get('test_prompt_path', str(default_prompt_path)))
    try:
        with open(prompt_path, "r", encoding='utf-8') as f:
            test_prompt = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Test prompt file not found at: {prompt_path}")

    # Evaluation
    testset_ids, hyps, refs = [], [], []
    for samples in tqdm(dataloader):
        # 디버깅 코드 추가
        print("Current batch testset_ids:", samples["testset_id"])
        print("Current batch size:", len(samples["testset_id"]))
        
        testset_id = samples["testset_id"]
        testset_ids.extend(testset_id)

        # Preprocess
        samples = prepare_sample(samples, cuda_enabled=torch.cuda.is_available())
        batch_size = samples["spectrogram"].shape[0]
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)
        speech_embeds, speech_atts = salmonn_preprocessor.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # Add prompt embeds + audio embed 
        prompts = [test_prompt[task] for task in samples['task']]
        templated_prompts = [cfg.config.model.prompt_template.format(prompt) for prompt in prompts]

        speech_embeds, speech_atts = salmonn_preprocessor.prompt_wrap(speech_embeds, speech_atts, templated_prompts, multi_prompt=True)
        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * tokenizer.bos_token_id

        bos_embeds = llama_model.get_input_embeddings()(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)

        generate_cfg = cfg.config.generate

        # Generation
        outputs = llama_model.generate(
            inputs_embeds=embeds,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
        )

        results = tokenizer.batch_decode(outputs)
        hyp = [result.split(generate_cfg.end_sym)[0].lower() for result in results]
        hyps.extend(hyp)

        if not args.skip_scoring:
            ref = samples.get("text", [""] * len(samples["testset_id"]))
            refs.extend(ref)

    if not args.skip_scoring:
        if args.task == 'asr':
            compute_wer(hyps, refs)
            
        elif args.task == 'aac':
            compute_spider(hyps, refs)

    output_dir = Path(cfg.config.run.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    task_name = args.task  # 'asr' or 'aac'
    output_file = output_dir / f"submission_{task_name}.csv"
    result_df = pd.DataFrame({"testset_id": testset_ids, "text": hyps})
    result_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    args = parse_args()

    random.seed(42)
    main(args)
