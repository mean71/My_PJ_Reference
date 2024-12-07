# src/data/data_loader.py
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Tuple

from src.data.dataset import ImageDataset, TextDataset  # 데이터셋 클래스 임포트
from src.data.preprocessing import preprocess_image, preprocess_text  # 전처리 함수 임포트
from utils.utils import good

def load_data(file_path: str) -> pd.DataFrame: # 주어진 파일경로에서 데이터를 로드
    """주어진 파일 경로에서 데이터를 로드합니다.
    :param file_path: str, 데이터 파일 경로
    :return: DataFrame 또는 list, 로드된 데이터
    :raises ValueError: 지원하지 않는 파일 형식일 경우 발생
    """
    if not os.path.isfile(file_path):  # 파일 존재 여부 확인
        raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")  # 파일이 없을 경우 예외 발생
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.txt'):
        return load_txt_data(file_path)
    elif file_path.endswith(('.jpg', '.jpeg', '.png')):
        return load_image_data(file_path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {file_path}")  # 지원하지 않는 형식 예외

def load_txt_data(file_path: str) -> pd.DataFrame:  # 텍스트 파일을 로드
    """텍스트 파일 로드 함수"""  # 텍스트 파일의 내용을 DataFrame으로 변환
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    data = [line.strip().split('\t') for line in lines]
    return pd.DataFrame(data[1:], columns=data[0])  # 첫 번째 줄을 헤더로 설정하여 DataFrame 반환

def load_image_data(file_path: str) -> Image.Image:  # 이미지 파일을 로드
    """이미지 파일 로드 함수"""  # PIL을 사용하여 이미지를 반환
    return Image.open(file_path)  # 이미지 객체 반환

def create_dataloader(file_path, batch_size, dataset_type):
    """주어진 파일에서 데이터를 로드하고 DataLoader를 생성합니다.
    
    :param file_path: str, 데이터 파일 경로
    :param batch_size: int, 배치 크기
    :param dataset_type: str, 데이터셋 유형 (예: 'model_a', 'model_b')
    :return: DataLoader, 배치 데이터 로더
    """
    data = load_data(file_path)  # 데이터 로드
    
    # 전처리 및 데이터셋 생성
    if dataset_type == 'model_a':
        features, labels = preprocess_model_a(data)
        dataset = CustomDatasetA(features, labels)
    elif dataset_type == 'model_b':
        features, labels = preprocess_model_b(data)
        dataset = CustomDatasetB(features, labels)
    else:
        raise ValueError("지원하지 않는 데이터셋 유형입니다.")

    # DataLoader 반환
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)






if __name__ == "__main__":
    file_path = "../data/processed/train_data.csv"  # 전처리된 데이터 경로
    data = load_data(file_path)                     # 데이터 로드
    dataset_type = 'model_a'                        # 데이터셋 유형 설정
    dataset = preprocess_data(data, dataset_type)   # 데이터 전처리
    print("dataset size:", len(dataset))
    
if __init__ == "__main__":
    pass