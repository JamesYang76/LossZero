import os
import torch
from mmengine import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules

# 모든 mmsegmentation 모듈 등록
register_all_modules()

# ------------------------------------------------------------------------------
# 1. ⚙️ 데이터셋 및 환경 설정 (from seonho_segmentation.ipynb)
# ------------------------------------------------------------------------------
DATA_DIR = os.path.expanduser("~/Projects/LossZero/data/Motorcycle Night Ride Dataset")
JSON_PATH = os.path.join(DATA_DIR, "COCO_motorcycle (pixel).json")
IMG_DIR = os.path.join(DATA_DIR, "images")

# 클래스 정의 (seonho 변수 순서: Rider, My bike, Moveable, Lane Mark, Road, Undrivable)
classes = ('Rider', 'My bike', 'Moveable', 'Lane Mark', 'Road', 'Undrivable')
palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [128, 128, 128]]

# ------------------------------------------------------------------------------
# 2. 📝 MMSegmentation 구성 (Config) 정의
# ------------------------------------------------------------------------------
# 추천 조합: DeepLabV3+ (모델) + ResNet50 (백본) + Weighted Loss + AMP (가속)

cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb2-40k_cityscapes-512x1024.py')

# 데이터셋 경로 및 타입 설정
cfg.dataset_type = 'CocoDataset'
cfg.data_root = DATA_DIR

# 모델 구조 수정 (클래스 수 6개로 변경)
cfg.model.decode_head.num_classes = 6
cfg.model.auxiliary_head.num_classes = 6

# 백본 및 가중치 설정 (Best Selection: ResNet50 + ImageNet Pretrained)
cfg.model.backbone.type = 'ResNet'
cfg.model.backbone.depth = 50
cfg.model.backbone.init_cfg = dict(type='Pretrained', checkpoint='torchvision://resnet50')

# 손실 함수 설정 (from joonwhan: Weighted CrossEntropy)
# 순서: [Rider: 2.0, My bike: 2.0, Moveable: 4.0, Lane Mark: 8.0, Road: 1.0, Undrivable: 1.0]
class_weights = [2.0, 2.0, 4.0, 8.0, 1.0, 1.0]
cfg.model.decode_head.loss_decode = dict(
    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=class_weights)

# 파이프라인 및 데이터 로더 설정
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

cfg.train_dataloader.dataset.type = cfg.dataset_type
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.ann_file = JSON_PATH
cfg.train_dataloader.dataset.data_prefix = dict(img_path='images', seg_map_path='labels') # 라벨 경로는 프로젝트 구조에 맞게 조정 필요
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
cfg.train_dataloader.batch_size = 4

# 스케줄러 및 옵티마이저 (from joonwhan: Adam 1e-4)
cfg.optim_wrapper.optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0001)

# 학습 가속 기술 (from jinkyu: Mixed Precision AMP)
cfg.optim_wrapper.type = 'AmpOptimWrapper'
cfg.optim_wrapper.loss_scale = 'dynamic'

# 체크포인트 및 로그 설정
cfg.work_dir = './work_dirs/byounggue_segmentation'
cfg.train_cfg.max_iters = 20000
cfg.default_hooks.checkpoint.interval = 5000
cfg.visualizer.vis_backends = [dict(type='LocalVisBackend'), dict(type='WandbVisBackend', init_kwargs=dict(project='LossZero'))]

# ------------------------------------------------------------------------------
# 3. 🚀 실행 (Runner)
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    runner = Runner.from_cfg(cfg)
    runner.train()
