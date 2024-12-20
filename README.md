### 프로젝트 구조 설명

```
project_name/
│
├── config/              # 설정 파일 폴더 (하이퍼파라미터, 경로 등)
│   └── config.yaml      # 공통 설정 파일
│
├── data/                # 데이터셋 저장 폴더
│   ├── raw/             # 원본 데이터 폴더
│   │   └── test_data_1/ # 원본 데이터 파일
│   └── processed/       # 전처리된 데이터 폴더
│       ├── dataset_name1/   # 데이터셋 1 관련 폴더
│       │   ├── test_data/
│       │   │   └── test_data_1.csv
│       │   └── train_data/
│       │       ├── train_data_1.csv
│       │       └── valid_data_1.csv
│       └── dataset_name2/   # 데이터셋 2 관련 폴더
│           ├── test_data/
│           │   └── test_data_1.csv
│           └── train_data/
│               ├── train_data_1.csv
│               └── valid_data_1.csv
│
├── models/              # 모델 저장 폴더 (학습된 모델 또는 다양한 모델 정의)
│   ├── checkpoints/     # 체크포인트 및 학습된 모델 파일
│   ├── base_model.py    # 기본 모델 아키텍처 정의 (상속용)
│   ├── model_a/         # 모델 A 관련 폴더
│   │   ├── checkpoints/  # 모델 A 체크포인트
│   │   ├── model.py      # 모델 A 아키텍처
│   │   └── config.yaml    # 모델 A 설정 파일
│   └── model_b/         # 모델 B 관련 폴더
│       ├── checkpoints/   # 모델 B 체크포인트
│       ├── model.py       # 모델 B 아키텍처
│       └── config.yaml     # 모델 B 설정 파일
│
├── src/                 # 주요 코드 폴더 (데이터 처리, 학습 등)
│   ├── data/            # 데이터 관련 코드
│   │   ├── dataset.py   # 데이터셋 클래스 정의
│   │   ├── dataset_factory.py  # 데이터셋 팩토리 클래스
│   │   ├── data_loader.py # 데이터 로딩 및 전처리 코드
│   │   └── preprocessing.py # 데이터 전처리 함수
│   │
│   ├── training/        # 학습 관련 코드
│   │   ├── train.py     # 모델 학습 코드
│   │   ├── evaluate.py  # 평가 코드
│   │   ├── metrics.py   # 평가 지표 관련 함수
│   │   └── checkpoint.py # 체크포인트 관리 코드
│   │
│   ├── inference/       # 추론 관련 코드
│   │   └── infer.py     # 추론 코드
│   │
│   ├── utils/           # 유틸리티 함수
│   │   ├── helpers.py    # 유용한 함수들 모음
│   │   ├── visualizer.py  # 시각화 관련 코드
│   │   └── logger.py      # 로그 관련 코드
│   │
│   ├── config.py        # 설정 관련 클래스
│   └── main.py          # 메인 실행 파일
│
├── notebooks/           # Jupyter 노트북 파일 저장 폴더 (데이터 분석, 실험 등)
│   └── EDA.ipynb        # 데이터 분석(EDA)용 노트북
│
│
├── logs/                # 로그 파일 폴더 (훈련 과정, 평가 결과 등 기록)
│   ├── training.log     # 학습 과정에서의 로그 파일
│   └── metrics.log      # 성능 지표 기록
│
├── tests/               # 테스트 코드 저장 폴더
│   └── test_model.py    # 모델의 단위 테스트
│
├── README.md            # 프로젝트 설명 파일
├── requirements.txt     # 의존성 라이브러리 목록
└── setup.py             # 패키지 설치 스크립트
```

1. 원본 데이터 로드: data/raw/test_data_1/ → src/data/data_loader.py
2. 데이터 가공 및 전처리: src/data/data_loader.py → src/data/preprocessing.py
3. 전처리 완료 데이터 저장: src/data/preprocessing.py → data/processed/
4. 모델 학습: data/processed/ → src/training/train.py
5. 모델 평가: src/training/evaluate.py → logs/metrics.log
6. 새로운 데이터 예측: src/inference/infer.py → logs/metrics.log
7. 유틸리티 제공: src/utils/
8. 학습 로그 기록: logs/training.log

- requirements.txt: 필요한 라이브러리 목록을 관리하여, 패키지 설치를 쉽게 합니다.
	- 필요한 라이브러리 파일과 버전을 명시적으로 설치하기 위한 .txt
	- cmd에서 다음 명령어를 실행하여 일괄 설치가능  ->  pip install -r requirements.txt

- config/: 하이퍼파라미터와 경로 설정 등 프로젝트에 필요한 설정 파일을 저장합니다.

- data/: 데이터셋을 저장하는 폴더로, 원본(raw) 데이터와 전처리된(processed) 데이터를 나누어 관리합니다.
  - raw/:              원본 데이터

- processed/:      전처리된 데이터
  - test_data/
  - train_data/
- models/: 모델 관련 파일을 저장하는 곳으로, 학습된 모델 체크포인트와 모델 아키텍처 정의 파일을 포함합니다. 여러 모델을 사용할 경우 각 모델별로 하위 폴더를 두어 관리할 수 있습니다.

- src/: 프로젝트의 주요 코드가 포함되는 폴더입니다. 데이터 로딩, 학습, 평가 및 추론과 관련된 스크립트가 위치합니다.
- data/
  - dataset.py   # 데이터셋 클래스 정의
    	- 데이터를 로드, 모델에 맞는 형식으로 변환 클래스 모음
  - dataset_factory.py  # 데이터셋 팩토리 클래스
    	- 
  - data_loader.py # 데이터 로딩 및 전처리 코드
  - preprocessing.py
- notebooks/: Jupyter 노트북 파일을 저장하는 폴더로, 데이터 분석이나 실험을 기록하는 데 유용합니다.

- logs/: 학습 과정에서 생성된 로그 파일을 저장하여, 모델의 훈련 과정을 추적할 수 있습니다.

- utils/: 유틸리티 함수를 모아놓은 폴더로, 반복적으로 사용되는 헬퍼 함수들이 포함됩니다.

- tests/: 모델의 단위 테스트 코드가 저장되어 있어, 코드의 품질을 보장할 수 있습니다.

- main.py: 프로젝트의 메인 실행 파일로, 모든 작업을 시작하는 진입점입니다.

- README.md: 프로젝트에 대한 설명, 사용법, 설치 방법 등을 문서화하는 파일입니다.

- setup.py: 패키지 설치를 위한 스크립트로, 프로젝트를 배포할 때 유용합니다.




# 데이터 설명
- 출처 및 설명

# 평가 지표

-  사용된 평가 지표 및 의미