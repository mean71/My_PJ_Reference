import yaml  # YAML 파일을 읽기 위한 라이브러리
import os    # 파일 경로를 다루기 위한 라이브러리

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 절대 경로
CONFIG_PATH = os.path.join(BASE_DIR, '../config.yaml') # 설정 파일 경로

def load_config(config_path):
    """
    설정 파일을 로드하는 함수.
    
    :param config_path: str, 설정 파일 경로
    :return: dict, 로드된 설정 내용
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)  # YAML 파일을 파싱하여 딕셔너리로 변환
    return config

if __name__ == "__main__":
    # 기본 설정 파일 경로
    config_file_path = "../config.yaml"  # 상대 경로 설정
    try:
        config = load_config(config_file_path)  # 설정 파일 로드
        print("설정 파일 내용:", config)  # 설정 내용 출력
    except Exception as e:
        print("오류 발생:", e)
