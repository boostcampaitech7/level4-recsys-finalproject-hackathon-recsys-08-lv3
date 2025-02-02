import torch
from models.salmonn import SALMONN
import numpy as np
import logging
from datasets import load_your_dataset  # 실제 데이터셋 로딩 함수

def load_pretrained_model(checkpoint_path):
    """
    사전 훈련된 모델 체크포인트 로드
    
    Args:
        checkpoint_path (str): 모델 체크포인트 경로
    
    Returns:
        SALMONN: 로드된 모델
    """
    try:
        # 체크포인트 로드
        ckpt = torch.load(checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화 및 가중치 로드
        model = SALMONN(
            llama_path="path/to/vicuna-7b",  # Vicuna 모델 경로 지정
            # 다른 필요한 설정들...
        )
        model.load_state_dict(ckpt['model'], strict=False)
        
        logging.info(f"Model loaded from {checkpoint_path}")
        return model
    
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise

def run_layer_pruning_experiment(model, max_layers, dataset):
    """
    Layer pruning 실험 수행
    
    Args:
        model (SALMONN): 사전 훈련된 모델
        max_layers (int): 모델의 최대 레이어 수
        dataset (list): 실험에 사용할 데이터셋
    
    Returns:
        dict: 각 레이어 수별 성능 결과
    """
    results = {}
    
    for num_layers in range(1, max_layers + 1):
        logging.info(f"Experimenting with {num_layers} layers")
        
        # 모델 레이어 pruning
        pruned_model = model.llama_model.model.prune_layers(num_layers)
        model.llama_model.model = pruned_model
        
        # 각 레이어 수에 대해 실험
        losses = []
        
        for batch in dataset:
            result = model.forward(batch, verbose=True)
            losses.append(result['loss'].item())
        
        results[num_layers] = {
            'avg_loss': np.mean(losses),
            'std_loss': np.std(losses),
            'losses': losses
        }
    
    return results

def main():
    # 로깅 설정
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 체크포인트 경로 (stage2 훈련 후 저장된 모델)
    checkpoint_path = "path/to/your/stage2/checkpoint.pth"
    
    # 모델 로드
    model = load_pretrained_model(checkpoint_path)
    
    # 데이터셋 로드 (실제 데이터셋 로딩 함수로 대체)
    dataset = load_your_dataset()
    
    # 최대 layer 수 (모델의 총 layer 수)
    max_layers = model.llama_model.model.config.num_hidden_layers
    
    # 실험 실행
    pruning_results = run_layer_pruning_experiment(model, max_layers, dataset)
    
    # 결과 출력 및 저장
    for layers, result in pruning_results.items():
        print(f"Layers: {layers}, Average Loss: {result['avg_loss']:.4f} ± {result['std_loss']:.4f}")
    
    # 결과를 JSON이나 CSV로 저장하는 코드 추가 가능
    # save_results_to_file(pruning_results)

if __name__ == "__main__":
    main()
