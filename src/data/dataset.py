# src/data/dataset.py
import os
import pandas as pd
from torch.utils.data import Dataset

class NameDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        """
        초기화 메서드
        :param data: 데이터프레임 또는 데이터 리스트
        :param labels: 라벨 (선택적)
        :param transform: 데이터 변환 함수 (선택적)
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self): # 데이터셋 크기 반환
        return len(self.data)

    def __getitem__(self, idx):
        """
        주어진 인덱스에 해당하는 데이터 및 라벨 반환
        :param idx: 데이터 인덱스
        :return: 데이터와 라벨 (변환이 적용될 경우 변환된 데이터)
        """
        item = self.data.iloc[idx]

        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        
        if self.transform:
            item = self.transform(item)

        return item, label




'''
데이터 형식:
이미지 데이터셋: 이미지 파일 경로를 읽고, 전처리(리사이징, 정규화 등)하는 기능이 필요.
텍스트 데이터셋: 텍스트 파일을 읽고, 토큰화, 패딩 등을 처리하는 기능이 필요.

라벨 처리:
지도 학습 데이터셋: 라벨이 있는 데이터를 처리하는 방식.
비지도 학습 데이터셋: 라벨 없이 데이터를 클러스터링하거나 특징을 추출하는 방식.

데이터 분할:
학습, 검증, 테스트 데이터셋: 각 데이터셋을 나누고, 이를 쉽게 로드할 수 있는 기능.
K-겹 교차 검증: K-겹 교차 검증을 위한 데이터셋 분할 기능.

전처리 기능:
각 데이터셋의 특성에 맞는 전처리 기능.
예: 표준화, 원-핫 인코딩, 피쳐 선택 등을 선택적으로 적용할 수 있는 옵션.

유연성 및 확장성:
인자화된 초기화: 다양한 데이터셋을 처리할 수 있도록 초기화 시 파라미터를 받아 처리할 수 있는 기능.
상속 가능성: 기본 클래스를 상속받아 다른 데이터셋을 처리할 수 있는 구조.

메타데이터 지원:
데이터셋에 대한 설명(예: 클래스 수, 샘플 수 등)을 제공하는 메타데이터 기능.

다양한 데이터 출처:
로컬 파일, 데이터베이스, API 등 다양한 출처에서 데이터를 로드할 수 있는 기능.
'''