# 데이터 관련 모듈
from src.data.data_loader import load_data
from src.data.preprocessing import preprocess_data
from src.data.dataset import CustomDataset
# 모델 관련 모듈
from models.model_a.model import ModelA  # 또는 ModelB
from models.checkpoints import load_checkpoint, save_checkpoint
# 훈련 관련 모듈
from src.training.train import train_model
from src.training.evaluate import evaluate_model
from src.training.metrics import accuracy, f1_score
# 추론 관련 모듈
from src.inference.infer import infer_model
# 유틸리티 모듈
from src.utils.helpers import setup_logger, get_config
# 설정 파일
from src.config import load_config

with open("config/config.yaml", 'r') as config_file:
    config = yaml.safe_load(config_file)

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def prepare_data(config):
    raw_data_path = config['data']['raw_data_path']
    processed_data_path = config['data']['processed_data_path']
    batch_size = config['models']['model_a']['batch_size']
    data_loader = DataLoader(raw_data_path, processed_data_path, batch_size)
    return data_loader.load_and_preprocess_data()

def prepare_model(config):
    model_config = config['models']['model_a']
    learning_rate = model_config['learning_rate']
    # 옵티마이저 정의는 여기서 처리
    # optimizer = ... (옵티마이저 초기화 코드)
    return learning_rate

def main():
    # 1. 설정 파일 로드
    config = load_config('config/config.yaml')
    
    test_data_path = config['data']['test_data_path']
    
    
    num_epochs = model_config['num_epochs']
    # 2. 데이터 로드 및 전처리, 분할
    data_loader = DataLoader(raw_data_path, processed_data_path, batch_size)
    train_data, valid_data = data_loader.load_and_preprocess_data()
    # 3. 모델 로딩
    model = define_model()
    train_data, valid_data = data_loader.load_and_preprocess_data()
    # 4. 모델 학습
    model = train_model(train_data, learning_rate, num_epochs)
    # 5. 모델 평가
    evaluation_results = evaluate_model(model, valid_data)
    print("Evaluation Results:", evaluation_results)
    # 6. 체크포인트 저장
    
    # 7. 모델 추론 (선택적)
    inference_results = infer_model(model, test_data)
    print("Inference Results:", inference_results)
    
    
    # 명령행 인자 파서 생성
    parser = argparse.ArgumentParser(description="AI 프로젝트 메인 실행 파일")
    
    # 서브 명령어 정의: train, evaluate, infer
    subparsers = parser.add_subparsers(dest="command")
    
    # 학습 명령어
    train_parser = subparsers.add_parser('train', help='모델 학습 실행')
    train_parser.add_argument('--model', type=str, required=True, choices=['model_a', 'model_b'], help='모델 선택')
    train_parser.add_argument('--config_path', type=str, required=True, help='설정 파일 경로')
    
    # 평가 명령어
    eval_parser = subparsers.add_parser('evaluate', help='모델 평가 실행')
    eval_parser.add_argument('--model', type=str, required=True, choices=['model_a', 'model_b'], help='모델 선택')
    eval_parser.add_argument('--checkpoint_path', type=str, required=True, help='체크포인트 파일 경로')
    
    # 추론 명령어
    infer_parser = subparsers.add_parser('infer', help='모델 추론 실행')
    infer_parser.add_argument('--model', type=str, required=True, choices=['model_a', 'model_b'], help='모델 선택')
    infer_parser.add_argument('--checkpoint_path', type=str, required=True, help='체크포인트 파일 경로')
    infer_parser.add_argument('--data_path', type=str, required=True, help='입력 데이터 경로')
    
    # 인자 파싱
    args = parser.parse_args()
    
    # 명령어에 따라 해당 기능 실행
    if args.command == 'train':
        train.main(args.model, args.config_path)
    elif args.command == 'evaluate':
        evaluate.main(args.model, args.checkpoint_path)
    elif args.command == 'infer':
        infer.main(args.model, args.checkpoint_path, args.data_path)

if __name__ == "__main__":
    # main()
    config = load_config()  # 기본 설정 파일 로드

    print("훈련 데이터 경로:", config['data']['train'])  # 훈련 데이터 경로 출력
    print("모델 타입:", config['model']['type'])
