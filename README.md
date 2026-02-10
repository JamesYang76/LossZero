# 🏍️ LossZero: Motorcycle Night Ride Semantic Segmentation

[![W&B](https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white)](https://wandb.ai/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/datasets/sadhliroomyprime/motorcycle-night-ride-semantic-segmentation)

**LossZero**는 야간 도로 주행 환경에서의 안전 확보를 위해 오토바이 주행 이미지를 활용한 **Multi-class Semantic Segmentation** 프로젝트입니다. 충돌 위험이 높은 도로 상황을 고려하여 정밀한 세그멘테이션 성능 확보와 결과의 해석 가능성(Interpretability)에 중점을 둡니다.

## 📌 Project Overview

상대적으로 데이터셋의 규모가 작은 야간 주행 환경 데이터를 활용하여 도로 위의 다양한 객체를 식별합니다. 특히 야간 가시성 확보와 객체 판별의 정확도가 생명과 직결되는 만큼, 고성능 모델 구축뿐만 아니라 모델의 판단 근거를 시각화하고 실험 과정을 철저히 관리합니다.

### Key Objectives
- **Robust Semantic Segmentation**: 야간 도로 주행 이미지에서 도로, 차량, 오토바이 등 다중 클래스를 정확하게 분할.
- **Explainable AI (XAI)**: CAM(Class Activation Map)을 통한 모델의 의사결정 시각화.
- **Strict Evaluation Metrics**: 단순 정확도를 넘어 도로 주행 안전성에 최적화된 상세 지표 수립.
- **Experiment Tracking**: W&B(Weights & Biases)를 이용한 하이퍼파라미터 및 실험 결과의 체계적 관리.

## 📊 Dataset

- **Source**: [Motorcycle Night Ride Semantic Segmentation (Kaggle)](https://www.kaggle.com/datasets/sadhliroomyprime/motorcycle-night-ride-semantic-segmentation)
- **Characteristics**: 야간 오토바이 주행 시점의 고해상도 이미지 및 클래스별 Segment Mask.

## 🛠️ Methodology & Tech Stack

### Architecture
- **Model**:  Segmentation Models (SAM, DeepLabV3+, U-Net++, or SegFormer 등 검토 중)
- **Framework**: PyTorch

### Performance Metrics (Safety-First)
도로 주행의 특수성을 고려하여 다음과 같은 지표를 상세히 모니터링합니다:
- **mIoU (mean Intersection over Union)**: 전체적인 클래스 분할 성능.
- **Pixel Accuracy**: 전체 픽셀 대비 정확도.
- **Class-wise IoU**: 각 클래스별(특히 위험 요소) 개별 성능 분석.
- **Boundary IoU**: 객체의 경계선(Boundary) 정밀 판독 능력 측정.

### Visualization & Management
- **CAM (Class Activation Mapping)**: 전역 평균 풀링(GAP) 혹은 Grad-CAM을 사용하여 모델이 특정 클래스로 판단할 때 주목한 영역을 히트맵으로 시각화.
- **W&B Integration**: 
  - 학습 곡선(Loss, Accuracy) 실시간 모니터링.
  - 하이퍼파라미터 스윕(Sweep)을 통한 최적의 조합 탐색.
  - 학습된 모델 체크포인트 관리 및 데이터 버전 관리.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch
- W&B Account

### Installation
```bash
git clone https://github.com/JamesYang76/LossZero.git
cd LossZero
```

## 📁 Project Structure

```text
LossZero/
├── data/                                   # Symbolic link or renamed data directory
│   └── Motorcycle Night Ride Dataset/      # Raw Dataset (from Kaggle)
│       ├── COCO_motorcycle (pixel).json    # Annotation file
│       └── images/                         # Source images and mask files
├── motorcycle.ipynb                        # Main development notebook
├── .gitignore
└── README.md
```

---
*Safe Riding through AI Precision — LossZero.*
