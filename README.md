# H형강 표면결함 탐지 시스템 (PatchCore)

H형강 제조 라인의 표면결함을 실시간으로 탐지하는 비지도 학습 기반 AI 시스템입니다.  
[PatchCore](https://arxiv.org/abs/2106.08265) (CVPR 2022) 알고리즘을 사용하며, **정상 이미지만으로 학습**합니다 — 결함 라벨링이 필요 없습니다.

## 핵심 특징

- **타일 기반 검출**: 1920x1200 원본 이미지를 256x256 타일로 분할하여 미세결함 디테일 보존
- **규격별 개별 모델**: 53규격 x 5카메라그룹 = 265개 전용 모델
- **Self-validation**: 라벨 없이 3라운드 반복 정제로 결함 혼입 데이터 자동 제거
- **Ens-MAX 앙상블**: 8개 독립 통계 지표의 z-score MAX 판정으로 강건한 결함 판단
- **빠른 규격 전환**: CNN 백본(WideResNet-50) 공유, 메모리뱅크만 교체하여 즉시 대응

## 처리 흐름

```
H형강 이미지 (1920x1200)
    |
    v
타일 분할 (256x256 패치)
    |
    v
피처 추출 (WideResNet-50, layer 2+3)
    |
    v
메모리뱅크(coreset) 대비 kNN 거리 계산
    |
    v
이상치 점수맵 -> Ens-MAX 판정
```

## 프로젝트 구조

```
src/
  config.py          # 설정 (NAS 경로, 카메라 그룹, 임계값)
  dataset.py         # NAS 데이터 로딩 및 RAM 프리로드
  patchcore.py       # PatchCore 핵심 구현
  self_validation.py # Self-validation 데이터 정제
  tile_mask.py       # 카메라 그룹별 동적 타일 마스킹
  utils.py           # 규격 탐색, 학습 가능 규격 필터링
train_v4_reorder.py  # 메인 학습 스크립트 (대형 규격 우선)
train_gpu1_reverse.py# GPU1 역순 학습 (듀얼 GPU 병렬)
inference.py         # 단일 이미지 추론 및 시각화
eval_all_v3.py       # 전체 규격 일괄 평가
monitor.py           # 학습 진행 모니터링
scan_nas.py          # NAS 폴더 스캐너
```

## 학습 방법

### 단일 규격
```bash
CUDA_VISIBLE_DEVICES=0 python train_v4_reorder.py --spec 700x300 --resume
```

### 전체 규격 (듀얼 GPU)
```bash
# GPU 0: 대형 규격부터 순차적으로
CUDA_VISIBLE_DEVICES=0 nohup python -u train_v4_reorder.py --all --resume >> train_gpu0.log 2>&1 &

# GPU 1: 소형 규격부터 역순으로
CUDA_VISIBLE_DEVICES=1 nohup python -u train_gpu1_reverse.py >> train_gpu1.log 2>&1 &
```

### 학습 결과물
```
output/{규격}/group_{1-5}/
  memory_bank.npy    # Coreset 피처
  threshold.json     # MAD 기반 임계값 (k=3.5)
  self_val_stats.json# Self-validation 통계
```

## 추론

```python
from src.patchcore import PatchCoreModel

model = PatchCoreModel("output/700x300/group_1")
score_map, max_score = model.predict(image)
```

## 카메라 그룹

| 그룹 | 카메라 | 촬영 부위 |
|------|---------|----------|
| 1 | 1, 10 | 웹 전면 |
| 2 | 2, 9 | 플랜지 상부 외면 |
| 3 | 3, 8 | 플랜지 상부 내면 |
| 4 | 4, 7 | 필릿 하부 |
| 5 | 5, 6 | 플랜지 하부 내면 |

## Self-Validation 정제 과정

1. **Round 0**: 전체 데이터로 학습, 이상치 점수 산출
2. **Round 1**: 고점수 타일 제외 (MAD 임계값, k=3.5), 재학습
3. **Round 2**: 최종 정제 및 메모리뱅크 확정

수동 라벨링 없이 학습 데이터 품질을 자동으로 확보합니다.

## 요구사항

- Python 3.10+
- PyTorch 2.0+ (CUDA)
- NVIDIA GPU 24GB+ VRAM (A40, L40S 테스트 완료)
- RAM 128GB+ (NAS 이미지 프리로드용)

## 참고문헌

- [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265) - Roth et al., CVPR 2022
- [WideResNet](https://arxiv.org/abs/1605.07146) - Zagoruyko & Komodakis, 2016

## 라이선스

MIT

## 저자

정상혁 (jsh@iae.re.kr)  
고등기술연구원 (IAE)
