import sys
import json
import torch
import librosa
import numpy as np
import soundfile as sf

from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import WhisperFeatureExtractor

# Add custom module path
sys.path.append(str(Path(__file__).parent / "audiolm-trainer"))

# Custom modules
from models.salmonn import SALMONN

def load_preprocessor(cfg):
    salmonn_preprocessor = SALMONN.from_config(cfg.config.model)
    
    # 디버깅용 코드
    print("Type of salmonn_preprocessor:", type(salmonn_preprocessor))
    if isinstance(salmonn_preprocessor, tuple):
        print("Tuple contents:")
        for i, item in enumerate(salmonn_preprocessor):
            print(f"Item {i}: {type(item)}")
    
    # 기존 코드 유지
    if isinstance(salmonn_preprocessor, tuple):
        salmonn_preprocessor = salmonn_preprocessor[0]  # 첫 번째 요소를 사용
    salmonn_preprocessor.to(cfg.config.run.device)
    salmonn_preprocessor.eval()
    return salmonn_preprocessor


def load_model(salmonn_preprocessor):
    model = salmonn_preprocessor.llama_model
    tokenizer = salmonn_preprocessor.llama_tokenizer
    return model, tokenizer


class SALMONNTestDataset(Dataset):
    def __init__(self, prefix, ann_path, whisper_path, task=None):
        super().__init__()

        self.prefix = prefix

        self.annotation = json.load(open(ann_path, "r"))["annotation"]

        self.wav_processor = WhisperFeatureExtractor.from_pretrained(whisper_path)

        self.task = task

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):        # 여러 샘플을 배치로 모으는 메서드
        samples_spectrogram = [s["spectrogram"] for s in samples]
        cat_spectrogram = torch.stack(samples_spectrogram, dim=0)

        raw_wav = [torch.from_numpy(s["raw_wav"]) for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        testset_id = [s["testset_id"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        entity = {
            "testset_id": testset_id,
            "spectrogram": cat_spectrogram,
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "task": task,
            "Q": Q,
            "id": id,
        }

        # text 키가 있는 경우에만 추가
        if any("text" in s for s in samples):
            entity['text'] = [s.get("text", "") for s in samples]

        return entity

    def __getitem__(self, index):   # 개별 데이터 샘플을 로드하고 전처리하는 메서드
        ann = self.annotation[index]
        audio_path = self.prefix + '/' + ann["path"]
        try:
            audio, sr = sf.read(audio_path)
        except:
            print(f"Failed to load {audio_path}. Load 0-th sample for now")
            audio, sr = sf.read(self.prefix + '/' + self.annotation[0]["path"])
        
        if len(audio.shape) == 2: # stereo to mono
            audio = audio[:, 0]

        if len(audio) < sr: # pad audio to at least 1s
            sil = np.zeros(sr - len(audio), dtype=float)
            audio = np.concatenate((audio, sil), axis=0)

        if sr != self.wav_processor.sampling_rate: # TODO. use more efficient implementation            
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.wav_processor.sampling_rate)
            sr = self.wav_processor.sampling_rate

        audio = audio[: sr * 30] # truncate audio to at most 30s

        spectrogram = self.wav_processor(audio, sampling_rate=sr, return_tensors="pt")["input_features"].squeeze()
        testset_id = ann["testset_id"]
        task = ann.get("task", "asr")
        Q = ann.get("Q", "")

        entity = {
            "testset_id": testset_id,
            "spectrogram": spectrogram,
            "raw_wav": audio,
            "task": task,
            "Q": Q,
            "id": ann["path"],
        }

        return entity

def collate_fn(batch):
    # 스펙트로그램 최대 길이 계산
    max_length = max(item['spectrogram'].shape[1] for item in batch)
    
    padded_spectrograms = []
    for item in batch:
        spec = item['spectrogram']
        if spec.shape[1] > max_length:
            # 자르기
            spec = spec[:, :max_length]
        else:
            # 패딩
            pad_width = max_length - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
        
        padded_spectrograms.append(spec)
    
    # 배치 항목 처리
    batch_items = {
        'spectrogram': torch.stack(padded_spectrograms),
        'testset_id': [item['testset_id'] for item in batch],
        'raw_wav': [item['raw_wav'] for item in batch],
        'task': [item['task'] for item in batch],
        'Q': [item.get('Q', '') for item in batch],
        'id': [item['id'] for item in batch]
    }

    # text 키가 있는 경우 추가
    if 'text' in batch[0]:
        batch_items['text'] = [item.get('text', '') for item in batch]

    return batch_items
