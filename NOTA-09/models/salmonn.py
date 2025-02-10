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

import logging
import json
import contextlib
import random
from awq import AutoAWQForCausalLM
from transformers import AutoModelForCausalLM
from awq.models.llama import LlamaAWQForCausalLM
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, WhisperModel
from peft import LoraConfig, TaskType, get_peft_model

from .Qformer import BertConfig, BertLMHeadModel
from .modeling_llama import LlamaForCausalLM
# from .modeling_whisper import WhisperModel
from .beats.BEATs import BEATsConfig, BEATs
from .utils import StoppingCriteriaSub

class LlamaAWQForCausalLMWrapper(LlamaAWQForCausalLM):
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            **kwargs,
        }

class SALMONN(nn.Module):
    #self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                 #   num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim
                #)
    @classmethod    # 인스턴스 생성 없이, 클래스에서 직접 호출 가능(SALMONN.~)
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        ## Qformer블록은 BertPredictionHeadTransform , 
        # BertLMPredictionHead, BertOnlyMsLMHead 참고
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    @property   # 모델이 어떤 디바이스(CPU, GPU)에서 실행되는지 알려주는 함수
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(device_type='cuda', dtype=dtype)
        else:
            return contextlib.nullcontext()

    def __init__(
        self,
        llama_path="",
        whisper_path="",
        freeze_whisper=True,
        beats_path="",
        freeze_beats=True,

        use_speech_Qformer=True,
        num_speech_query_token=1,
        freeze_speech_QFormer=False,
        window_level_Qformer=True,
        second_per_window=0.333333,
        second_stride=0.333333,
        
        speech_llama_proj_model="",
        freeze_speech_llama_proj=False,

        lora=True,
        lora_rank=8,
        lora_alpha=32,
        lora_dropout=0.1,

        multi_prompt=False,
        prompt_path="",
        prompt_template="",
        max_txt_len=128,
        end_sym="</s>",
        low_resource=False,  # use 8 bit
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        token=None,
        only_preprocessor=None,
        prune=False,
        awq="",
        whisper_config="",
        whisper_lora=False
    ):
        super().__init__()

        self.beats_path = beats_path # 비트 인코더의 state_dic
        self.use_speech_Qformer = use_speech_Qformer # q-former사용할지말지
        self.window_level_Qformer = window_level_Qformer # windoe level 몇개할지?
        self.second_per_window = second_per_window # window 별로 몇초정도로 자르는지
        self.second_stride = second_stride # 윈도우 와 윈도우 사이 간격
        self.lora = lora # lora  쓸꺼인지
        self.multi_prompt = multi_prompt # 한 배치안에 여러개의 psompt가 있는지
        self.max_txt_len = max_txt_len # 입력 테스트의 최대 갯수
        self.end_sym = end_sym  # "<|end_of_text|>" 텍스트의 종료를 알리는 입력끝표시
        self.low_resource = low_resource # quantizaion진행할지
        self.prune=prune
        self.awq=awq
        self.whisper_config=whisper_config
        self.whisper_lora=whisper_lora

        logging.info('Loading LLaMA Tokenizer')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_path, use_fast=False, token=token)  # LLama의 모델 경로 에서 토크나이저를 불러일으킴 (주로 모델이 텍스트를 토큰으로 변환하는데 쓰임)
        self.llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})  ### 패딩을 넣는곳에씀
        self.llama_tokenizer.padding_side = "right" ## 오른쪽.으로

        if not only_preprocessor:  # lamma model 불러오는거야
            logging.info('Loading LLaMA Model')
            if self.awq and self.low_resource:
                print("You cannot implement both awq and bitsandbytes")
                raise ValueError("Conflicting quantization methods: QAT and bitsandbytes cannot be used together.")
            elif self.low_resource:
                bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,  # ✅ Use 4-bit quantization instead of 8-bit
                        bnb_4bit_compute_dtype=torch.float16,  # ✅ Keep computations in FP16
                        bnb_4bit_quant_type="nf4",  # ✅ Normalized Float 4 (best for QLoRA)
                        bnb_4bit_use_double_quant=True  # ✅ Improves quantization accuracy
                    )
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_path,
                    quantization_config=bnb_config,
                    device_map={"": device_8bit},
                    token=token,
                )
            elif self.awq:
                self.llama_model = LlamaAWQForCausalLMWrapper.from_quantized(
                                    model_path=self.awq,
                                    model_type='llama',
                                    quant_filename="model.safetensors",
                                    max_seq_len=self.max_txt_len,
                                    torch_dtype=torch.float16,
                                    ddevice="cuda",
                                    trust_remote_code=True
                                )
            else:
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    llama_path,
                    torch_dtype=torch.float16,
                    token=token,
                )
            if not self.awq:
                self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
            else:
                self.llama_model.model.resize_token_embeddings(len(self.llama_tokenizer))
                
            ## 모델 얼리는거
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            logging.info('Loading LLaMA Done')
            # lora넣을꺼면 컨피크하는거 나중에 다시 알아보자
            if self.lora:
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, 
                    inference_mode=False, 
                    r=lora_rank, 
                    lora_alpha=lora_alpha, 
                    lora_dropout=lora_dropout,
                )
                
                self.llama_model = get_peft_model(self.llama_model, self.peft_config)
                self.llama_model.print_trainable_parameters()
                logging.info('LoRA Training')
                    
                
        #whisper_path넣기
        assert whisper_path
        logging.info('Loading Whisper Model')
        if 'H100' in torch.cuda.get_device_name(0):
            self.speech_encoder = WhisperModel.from_pretrained(whisper_path,torch_dtype=torch.float16,attn_implementation="flash_attention_2").encoder
        else:    
            self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder #(encoder + decoder 에서 encoder만 분리)
        self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model) #( encoder로 나온 represtation 들 다 노말라이즈시키기)
        if freeze_whisper: # 얼림
            for name, param in self.speech_encoder.named_parameters():
                param.requires_grad = False
            logging.info("freeze Whisper")
        
        if self.whisper_lora:
            self.whisper_lora_config = LoraConfig(
                        inference_mode=False,  # Enables training
                        r=lora_rank,  # Low-rank adaptation rank
                        lora_alpha=lora_alpha,  # Scaling factor
                        lora_dropout=lora_dropout,  # Regularization
                        target_modules=["q_proj", "v_proj", "o_proj"],  # Apply LoRA to attention layers
                    )
            self.speech_encoder=get_peft_model(self.speech_encoder, self.whisper_lora_config)
            print(self.speech_encoder)
            logging.info('Whisper LoRA Training')
        
        if self.beats_path: 
            logging.info("Loading BEATs Model")
            beats_ckpt = torch.load(self.beats_path, map_location='cpu', weights_only=True) # torch.load로 beat 훈련된거 불러오기
            beats_cfg = BEATsConfig(beats_ckpt['cfg']) # Beats config (attribution을 불러와 업데이트함)
            self.beats = BEATs(beats_cfg)
            self.beats.load_state_dict(beats_ckpt['model'])
            self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
            if freeze_beats:
                for name, param in self.beats.named_parameters():
                    param.requires_grad = False
                self.beats.eval()
                logging.info("freeze BEATs")

        # Q_former불르자
        if self.use_speech_Qformer:
            if self.beats_path:
                # beats 가 주어지면 speech_width (차원은  whisper와 beats둘다 차원이 합해져서 나옴)
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model + self.beats.cfg.encoder_embed_dim
                )
            else:
                self.speech_Qformer, self.speech_query_tokens = self.init_speech_Qformer(
                    num_query_token=num_speech_query_token, speech_width=self.speech_encoder.config.d_model
                )
            self.speech_Qformer.bert.embeddings.word_embeddings = None
            self.speech_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.speech_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.speech_Qformer.cls = None
            if freeze_speech_QFormer:
                for name, param in self.speech_Qformer.named_parameters():
                    param.requires_grad = False
                self.speech_Qformer.eval()
                self.speech_query_tokens.requires_grad = False
                logging.info("freeze Speech QFormer")

            logging.info('Loading speech LLAMA proj')
        
            if only_preprocessor:
                config = AutoConfig.from_pretrained(llama_path, token=token)
                lm_hidden_size = config.hidden_size
            else:
                lm_hidden_size = self.llama_model.config.hidden_size
            #
            self.speech_llama_proj = nn.Linear(
                self.speech_Qformer.config.hidden_size, lm_hidden_size
            )
            if speech_llama_proj_model:
                logging.info("Loading speech LLAMA proj from {}".format(speech_llama_proj_model))
                speech_llama_proj_weight = torch.load(speech_llama_proj_model, map_location="cpu")
                self.load_state_dict(speech_llama_proj_weight['model'], strict=False)
            if freeze_speech_llama_proj:
                for name, param in self.speech_llama_proj.named_parameters():
                    param.requires_grad = False
                self.speech_llama_proj.eval()
                logging.info("freeze speech LLAMA proj")
        else:
            # feel free to add other aligners here
            raise NotImplementedError

        # prepare prompts
        # prompt준비?..
        self.prompt_dict = {}
        if prompt_path:
            try:
                raw_prompts = json.load(open(prompt_path, "r"))
            except:
                print("Failed to load prompt! Try to use utf-8 encoding.")
                raw_prompts = json.load(open(prompt_path, "r", encoding='utf-8'))
            for task in raw_prompts.keys():
                filted_prompts = [raw_prompt for raw_prompt in raw_prompts[task] if "<SpeechHere>" in raw_prompt]
                self.prompt_dict[task] = [prompt_template.format(p) for p in filted_prompts]
            print("Loading training prompts done!")
        # self.prompot_dict[task] = ["USER: {filtered_prompt_for_task}\nASSISTANT:"]

    def apply_pruned_layers(self, pruned_layers):
        if self.awq:
            self.llama_model.base_model.model.model.model.layers=torch.nn.ModuleList([
                        layer for i, layer in enumerate(self.llama_model.base_model.model.model.model.layers) if i not in pruned_layers])
        else:
            self.llama_model.base_model.model.model.layers= torch.nn.ModuleList([
                        layer for i, layer in enumerate(self.llama_model.base_model.model.model.layers) if i not in pruned_layers])
        torch.cuda.empty_cache()
        logging.info("LLAMA MODEL Layer pruned")

    def apply_whisper_prune(self,pruned_layers):
        self.speech_encoder.base_model.model.layers=torch.nn.ModuleList([
                    layer for i, layer in enumerate(self.speech_encoder.base_model.model.layers) if i not in pruned_layers
                ])
        logging.info("Whisper Encoder Layer pruned")
        torch.cuda.empty_cache()
        

    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None):
        with self.maybe_autocast():
            #speech_embds (whisper)랑 + beats embedings 
            if self.use_speech_Qformer:
                speech_embeds = self.ln_speech(speech_embeds)
                if audio_embeds is not None:
                    audio_embeds = self.ln_audio(audio_embeds)
                    if audio_embeds.size(1) < speech_embeds.size(1):
                        audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                    elif audio_embeds.size(1) > speech_embeds.size(1):
                        speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                    speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
                    # (B,T,speech_embeds.size(2)+audio_embeds.size(2))
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
                # speech_atts -> (B,T)
                if self.window_level_Qformer:
                    B, T, C = speech_embeds.shape
                    kernel = round(1500 * self.second_per_window / 30.0)
                    stride = round(1500 * self.second_stride / 30.0)
                    kernel = (1, kernel)
                    stride = (1, stride)
                    speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
                    speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
                    _, _, L = speech_embeds_overlap.shape
                    speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
                    speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
                    speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
                    speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)
                # speech_embeds => (B*L, kernel[1], C)
                # speech_atts => (B*L,kernel[1])
                query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
                # query tokens (1,1,hidden size)였는데 바뀜. =>(B*L,1,hidden_size)
                # 왜 .bert에 넣는지 조금 이해가안됨? .bert는 자동으로 is_d
                query_output = self.speech_Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=speech_embeds,
                    encoder_attention_mask=speech_atts,
                    return_dict=True,
                )
                #query_output.last_hidden_state => #(B*L,kernel[1],all_head_size) # L => the number of windows 
                speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)

                if self.window_level_Qformer:
                    speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()

                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
            else:
                raise NotImplementedError

        return speech_embeds, speech_atts

    def encode_speech(self, spectrogram, raw_wav=None, audio_padding_mask=None):
        with self.maybe_autocast():
            #whisper -> speech_embds
            speech_embeds = self.speech_encoder(spectrogram, return_dict=True).last_hidden_state
            
            if self.beats_path and raw_wav is not None:
                audio_embeds, _ = self.beats.extract_features(raw_wav, padding_mask=audio_padding_mask, feature_only=True)
            else:
                audio_embeds = None
                        
        return self._encode_auditory_feature(speech_embeds, audio_embeds=audio_embeds)

    def prompt_wrap(self, embeds, atts, prompt, multi_prompt=False):
        if prompt:
            if multi_prompt:
                p_before = []
                p_after = []
                for i, p in enumerate(prompt):
                    b, a = p.split("<SpeechHere>") # <Speech><SpeechHere><Speech> Please describe ==> b=
                    p_before.append(b) 
                    p_after.append(a)
                
                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids)

                # speech_embeds wrapped with prompts_embeds are padded to the same length here
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", padding="longest", add_special_tokens=False
                ).to(embeds.device)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            else:
                batch_size = embeds.shape[0]
                p_before, p_after = prompt.split("<SpeechHere>")

                p_before_tokens = self.llama_tokenizer(
                    p_before, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_after_tokens = self.llama_tokenizer(
                    p_after, return_tensors="pt", add_special_tokens=False
                ).to(embeds.device)
                p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
                p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) if not self.lora else self.llama_model.model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

                wrapped_embeds = torch.cat([p_before_embeds, embeds, p_after_embeds], dim=1)
                wrapped_atts = torch.cat([p_before_tokens.attention_mask, atts, p_after_tokens.attention_mask], dim=1)
            return wrapped_embeds, wrapped_atts
        else:
            return embeds, atts
        
    def forward(self, samples, verbose=False):
        #트레이닝
        # detect whether there are multi tasks in this batch
        task = list(set(samples["task"]))
        if len(task) > 1 or "QA" in task:
            self.multi_prompt = True

        # prepare prompts
        if self.prompt_dict:
            if self.multi_prompt:
                prompt = [random.choice(self.prompt_dict[task]) for task in samples["task"]]
                if "Q" in samples:
                    prompt = [p.format(q) if '{}' in p else p for p, q in zip(prompt, samples["Q"]) ]
            else:
                prompt = random.choice(self.prompt_dict[samples["task"][0]])

        # use speech/audio encoder to encode speech/audio
        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)
        #encode_speech ? => ()
        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        # wrap speech_embeds with prompts
        if self.prompt_dict:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompt, multi_prompt=self.multi_prompt)

        # prepare inputs for LLM
        text = [t + self.end_sym for t in samples["text"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(spectrogram.device)
        # to_regress_embeds -> 정답
        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids) if not self.lora else self.llama_model.model.model.embed_tokens(to_regress_tokens.input_ids)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        empty_targets = (
            torch.ones(
                [speech_atts.shape[0], speech_atts.shape[1] + 1],
                dtype=torch.long
            ).to(spectrogram.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        # 타겟 생성 끝.
        #[empty,empty,empty     the dog, is , barking  ]
        #[<speech>,speech_emb,<speech_hear>,describe,this,sound,the dog, is barking]

        batch_size = speech_embeds.shape[0]
        bos = torch.ones(
            [batch_size, 1],
            dtype=to_regress_tokens.input_ids.dtype,
            device=to_regress_tokens.input_ids.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        inputs_embeds = torch.cat([bos_embeds, speech_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, speech_atts, to_regress_tokens.attention_mask], dim=1)

        # calulate loss
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss

        if verbose:
            nvocab = self.llama_model.config.vocab_size
            results = outputs.logits[:, empty_targets.size(1) - 1: -1, :].contiguous().view(-1, nvocab).argmax(dim=-1)
            labels = targets[:, empty_targets.size(1):].contiguous().view(-1)
            mask = (labels != -100)
            correct = (results[mask] == labels[mask]).float().sum()
            total = len(labels[mask])

        if verbose:
            return {"loss": loss, "correct": correct, "total": total}

        return {"loss": loss}
    # 추론
    def generate(self, samples, generate_cfg, prompts=None):
        batch_size = samples["spectrogram"].shape[0]

        spectrogram = samples["spectrogram"]
        raw_wav = samples.get("raw_wav", None)
        audio_padding_mask = samples.get("padding_mask", None)

        speech_embeds, speech_atts = self.encode_speech(spectrogram, raw_wav=raw_wav, audio_padding_mask=audio_padding_mask)

        if prompts is not None:
            speech_embeds, speech_atts = self.prompt_wrap(speech_embeds, speech_atts, prompts, multi_prompt=True)

        bos = torch.ones(
            [batch_size, 1],
            dtype=torch.int32,
            device=speech_embeds.device,
        ) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos) if not self.lora else self.llama_model.model.model.embed_tokens(bos)
        atts_bos = speech_atts[:, :1]

        embeds = torch.cat([bos_embeds, speech_embeds], dim=1)
        attns = torch.cat([atts_bos, speech_atts], dim=1)

        stop_words_ids = [torch.tensor([2]).to(speech_embeds.device)] # TODO: fix this heuristics  
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=generate_cfg.get("max_new_tokens", 200),
            stopping_criteria=stopping_criteria,
            num_beams=generate_cfg.get("num_beams", 4),
            do_sample=generate_cfg.get("do_sample", False),
            min_length=generate_cfg.get("min_length", 1),
            temperature=generate_cfg.get("temperature", 1.0),
            top_p=generate_cfg.get("top_p", 0.9),
            repetition_penalty=generate_cfg.get("repetition_penalty", 1.0),
            length_penalty=generate_cfg.get("length_penalty", 1.0),
            attention_mask=attns,
        )        
        text = self.llama_tokenizer.batch_decode(outputs, add_special_tokens=False)

        return text

    @classmethod
    def from_config(cls, config):
        llama_path = config.get("llama_path")
        whisper_path = config.get("whisper_path")
        freeze_whisper = config.get("freeze_whisper", True)
        beats_path = config.get("beats_path", "")
        freeze_beats = config.get("freeze_beats", True)

        use_speech_Qformer = config.get("use_speech_Qformer", True)
        num_speech_query_token = config.get("num_speech_query_token", 1)
        freeze_speech_QFormer = config.get("freeze_speech_QFormer", False)
        window_level_Qformer = config.get("window_level_Qformer", True)
        second_per_window = config.get("second_per_window", 0.333333)
        second_stride = config.get("second_stride", 0.333333)

        speech_llama_proj_model = config.get("speech_llama_proj_model", "")
        freeze_speech_llama_proj = config.get("freeze_speech_llama_proj", False)

        lora = config.get("lora", True)
        lora_rank = config.get("lora_rank", 8)
        lora_alpha = config.get("lora_alpha", 32)
        lora_dropout = config.get("lora_dropout", 0.1)

        multi_prompt = config.get("multi_prompt", False)
        prompt_path = config.get("prompt_path", "")
        prompt_template = config.get("prompt_template", "")
        max_txt_len = config.get("max_txt_len", 128)
        end_sym = config.get("end_sym", "</s>")
        low_resource = config.get("low_resource", False)
        device_8bit = config.get("device_8bit", 0)

        token = config.get("token", None)
        only_preprocessor = config.get("only_preprocessor", None)
        prune=config.get('prune',False)
        awq=config.get('awq',"")
        whisper_config = config.get("whisper_config","")
        whisper_lora= config.get("whisper_lora",False)

        model = cls(
            llama_path=llama_path,
            whisper_path=whisper_path,
            freeze_whisper=freeze_whisper,
            beats_path=beats_path,
            freeze_beats=freeze_beats,
            use_speech_Qformer=use_speech_Qformer,
            num_speech_query_token=num_speech_query_token,
            freeze_speech_QFormer=freeze_speech_QFormer,
            window_level_Qformer=window_level_Qformer,
            second_per_window=second_per_window,
            second_stride=second_stride,
            speech_llama_proj_model=speech_llama_proj_model,
            freeze_speech_llama_proj=freeze_speech_llama_proj,
            lora=lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            multi_prompt=multi_prompt,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            token=token,
            only_preprocessor=only_preprocessor,
            prune= prune,
            awq=awq,
            whisper_config=whisper_config,
            whisper_lora=whisper_lora
        )

        ckpt_path = config.get("ckpt", "")
        if ckpt_path:
            logging.info("Load SALMONN ckpt from: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            if model.prune and 'pruned_layers' in ckpt:
                ## prune 하자
                layer_to_prune=ckpt['pruned_layers']
                model.apply_pruned_layers(layer_to_prune)
            if model.prune and 'whisper_prune' in ckpt:
                layer_to_prune=ckpt['whisper_prune']
                model.apply_whisper_prune(layer_to_prune)
            model.load_state_dict(ckpt['model'], strict=False)
        return model
