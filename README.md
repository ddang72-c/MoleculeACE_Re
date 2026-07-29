# MoleculeACE_Re

[MoleculeACE](https://github.com/molML/MoleculeACE) 벤치마크 기준선을 돌리는 공통 코드.
새 방법을 만들었을 때 그 성적을 공식 결과표와 나란히 놓으려면, 먼저 내 코드가 공식과
같은 조건에서 돌아간다는 것을 보여야 한다.

## 범위 — 공식 24접근 중 **17개 재현 완료**

| 계열 | 구성 | 개수 | 상태 |
|---|---|--:|---|
| 전통 ML | 모델 4종(SVM·RF·GBM·KNN) × 표현 4종(ECFP·MACCS·PHYSCHEM·WHIM) | **16** | ✅ 완료 (480셀) |
| **MLP + ECFP** | 신경망 기준선 | **1** | ✅ 완료 (30셀) |
| 그래프 신경망 | GCN · GAT · MPNN · AFP | 4 | ⏸️ 진행 중 |
| SMILES 딥러닝 | LSTM · CNN · Transformer | 3 | ⏸️ 진행 중 |

**17 × 30표적 = 510셀.** 나머지 7종은 코드(`scripts/run_deep.py`)와 재현에 필요한 지식은
아래에 정리돼 있으나 **수치는 아직 확정하지 않았다** — 이유는 「비용」 절 참조.

## 결과 — 평균은 전부 재현된다

| 표현 | 모델 | RMSE 재현 | 공식 | 차 | cliff 재현 | 공식 | 차 | 셀 일치 | 최대 셀오차 |
|---|---|---:|---:|---:|---:|---:|---:|:--:|---:|
| ECFP | SVM | 0.6712 | 0.6712 | -0.0000 | 0.7511 | 0.7511 | -0.0000 | 30/30 | 0.0001 |
| ECFP | GBM | 0.6932 | 0.6948 | -0.0016 | 0.7870 | 0.7835 | +0.0036 | 0/30 | 0.0693 |
| ECFP | RF | 0.7009 | 0.7000 | +0.0009 | 0.7847 | 0.7857 | -0.0010 | 0/30 | 0.0236 |
| ECFP | KNN | 0.7292 | 0.7290 | +0.0003 | 0.8393 | 0.8388 | +0.0005 | 22/30 | 0.0062 |
| MACCS | GBM | 0.7490 | 0.7495 | -0.0005 | 0.8462 | 0.8416 | +0.0046 | 0/30 | 0.0973 |
| MACCS | SVM | 0.7520 | 0.7520 | -0.0000 | 0.8499 | 0.8499 | +0.0000 | 30/30 | 0.0001 |
| MACCS | RF | 0.7561 | 0.7557 | +0.0004 | 0.8471 | 0.8472 | -0.0001 | 0/30 | 0.0210 |
| MACCS | KNN | 0.8110 | 0.8110 | +0.0000 | 0.8999 | 0.9001 | -0.0002 | 22/30 | 0.0049 |
| PHYSCHEM | RF | 0.8848 | 0.8807 | +0.0041 | 0.9350 | 0.9302 | +0.0049 | 0/30 | 0.0503 |
| PHYSCHEM | GBM | 0.8935 | 0.8913 | +0.0022 | 0.9460 | 0.9464 | -0.0005 | 0/30 | 0.0802 |
| PHYSCHEM | KNN | 0.9176 | 0.9177 | -0.0001 | 0.9479 | 0.9433 | +0.0045 | 0/30 | 0.0523 |
| PHYSCHEM | SVM | 0.9348 | 0.9373 | -0.0025 | 0.9646 | 0.9636 | +0.0010 | 0/30 | 0.0476 |
| WHIM | RF | 0.9771 | 0.9763 | +0.0008 | 1.0187 | 1.0144 | +0.0043 | 0/30 | 0.1084 |
| WHIM | GBM | 0.9908 | 0.9946 | -0.0038 | 1.0396 | 1.0294 | +0.0102 | 0/30 | 0.1513 |
| WHIM | SVM | 1.0044 | 1.0083 | -0.0039 | 1.0578 | 1.0516 | +0.0062 | 0/30 | 0.1329 |
| WHIM | KNN | 1.0226 | 1.0162 | +0.0064 | 1.0574 | 1.0446 | +0.0128 | 1/30 | 0.0799 |

평균 차이는 16조합 전부 **0.0064 이내**다. 표적별 상세는
[`results/ml16_reproduction.csv`](results/ml16_reproduction.csv).

## ⚠️ 셀 단위로는 조합마다 갈린다 — 원인을 확인했다

`ECFP+SVM`과 `MACCS+SVM`만 30/30 정확히 맞고 나머지는 안 맞는다. **세 가지 원인이 겹쳐 있다.**

### ① 트리 앙상블에 시드가 없다 (RF · GBM)

`get_benchmark_config`가 `random_state`를 주지 않고 패키지의 `RANDOM_SEED = 42`도 적용되지
않는다. 같은 표적을 RF로 세 번 돌리면 `0.665992 / 0.658890 / 0.665117`이 나온다.
`--seed`로 고정할 수 있다.

### ② PHYSCHEM — `NumHAcceptors` 정의가 rdkit 버전 간 바뀌었다

PHYSCHEM은 순수 2D 기술자 11개다. 그중 **10개는 rdkit 2022.09.5와 2026.03.4에서 값이 같고,
`NumHAcceptors` 하나만 다르다.**

```
FC(F)(F)c1cccc(-c2nnc3ccc(NC4CCCCC4)cn23)c1
    rdkit 2022.09.5 → NumHAcceptors = 4
    rdkit 2026.03.4 → NumHAcceptors = 3
```

SVM은 결정적인데도 PHYSCHEM에서 0/30인 이유가 이것이다. **모델이 아니라 입력이 달라진다.**

### ③ WHIM — 시드가 있어도 컨포머가 달라진다

`EmbedMolecule(..., randomSeed=42)`로 시드가 박혀 있지만, **ETKDG 구현 자체가 버전 간
바뀌어서 같은 시드로도 다른 좌표가 나온다.** 같은 분자의 WHIM 114개 중 4개가 달라진다
(최대차 0.001). 표적 전체로 누적되면 셀 단위 불일치가 된다.

> **⇒ 정리:** numpy 2 때문에 rdkit을 올려야 하는데, **그 업그레이드가 PHYSCHEM과 WHIM 값을
> 바꾼다.** ECFP·MACCS는 비트 정의가 고정이라 영향이 없다.
> **셀 단위 대조가 필요하면 ECFP·MACCS + SVM 으로 하고, PHYSCHEM·WHIM 은 평균으로 비교하라.**

## 설치

```bash
git clone https://github.com/ddang72-c/MoleculeACE_Re
cd MoleculeACE_Re

uv venv --python 3.11 mace-env
uv pip install --python mace-env/bin/python MoleculeACE
uv pip install --python mace-env/bin/python torch torch_geometric transformers tensorflow
uv pip install --python mace-env/bin/python -U rdkit
```

네 줄을 순서대로 다 실행해야 한다.

- 패키지 `__init__.py`가 MPNN을 먼저 불러서 **SVM 하나만 써도 torch/PyG/transformers/
  tensorflow가 전부 필요하다.**
- **마지막 `-U rdkit`을 빼면 `AttributeError: _ARRAY_API not found`로 죽는다.**
  패키지가 끌어오는 rdkit 2022.09.5가 numpy 2와 충돌한다.

원 패키지 README가 요구하는 Python 3.8 / TF 2.9 / PyTorch 1.11은 **따를 필요 없다.**
위 수치는 Python 3.11, torch 2.13, PyG 2.8, transformers 5.14, rdkit 2026.3.4에서 나왔다.
데이터셋은 `pip install` 할 때 패키지 안에 같이 깔린다(7MB, 따로 받을 필요 없음).

## 실행

```bash
./mace-env/bin/python scripts/run_baselines.py    # 인자로 조합 선택
./mace-env/bin/python scripts/run_ml16.py         # 16조합 전부 (WHIM 때문에 1시간+)
```

`run_baselines.py` 옵션:

```bash
--models SVM RF            # 모델 고르기 (SVM RF GBM KNN MLP)
--descriptor MACCS        # 표현 고르기 (ECFP MACCS PHYSCHEM WHIM)
--targets CHEMBL204_Ki    # 표적 고르기
--seed 42                 # 시드 고정 (RF/GBM/MLP 에만 적용)
--out results/my.csv      # 출력 경로
```

하이퍼파라미터는 `get_benchmark_config`가 주므로 튜닝할 것이 없다.
SVM+ECFP는 30표적에 약 20초, WHIM은 컨포머 생성 때문에 표적당 1~8분 걸린다.

---

# 딥러닝 계열 — MLP 완료, 나머지 7종 진행 중 (2026-07-29)

`scripts/run_deep.py`

## 🚨 먼저 — 공식 딥러닝 프로토콜은 단일 학습이 아니다

공식 [`Experiments/benchmark.py`](https://github.com/molML/MoleculeACE/blob/main/Experiments/benchmark.py)를 보면
딥러닝 계열(RF·SVM·GBM·KNN 제외 전부)은 이렇게 학습된다.

```python
if descriptor in [Descriptors.SMILES, Descriptors.TOKENS]:
    data.augment(10); data.shuffle()          # 10배 증강
hyperparameters.pop("epochs")                 # epochs 를 버린다
rmse, cliff = cross_validate(algo, data, n_folds=5,
                             early_stopping=10, seed=RANDOM_SEED, ...)
rmse = sum(rmse)/len(rmse)                    # 5겹 평균
```

**5겹 교차검증 평균 + patience 10 조기종료 + SMILES/TOKENS 10배 증강.**
셋 중 하나라도 빠지면 재현이 아니라 다른 실험이다.
셋을 다 빼고 단일 학습했을 때 Transformer 가 0.514 로 공식(0.7317)보다 "좋게" 나온 적이 있는데,
개선이 아니라 비교 대상이 아니었던 것이다.

⚠️ **전통 ML 16조합은 영향 없다** — 공식도 단일 학습이므로 위 재현은 그대로 유효하다.

## 결과 — MLP + ECFP (30/30 완료)

| 모델 | RMSE 재현 | 공식 | 차 | cliff 재현 | 공식 | **절벽격차 내것** | **공식** |
|---|--:|--:|--:|--:|--:|--:|--:|
| **MLP + ECFP** | **0.7352** | 0.7632 | **-0.0280** | 0.8090 | 0.8333 | **0.0738** | 0.0701 |

표적별 상세는 [`results/mlp_ecfp_reproduction.csv`](results/mlp_ecfp_reproduction.csv).
전체 평균이 -0.028 로 맞고, **절벽 격차(0.0738 vs 공식 0.0701)까지 재현된다.**

⏸️ **나머지 7종(GCN·GAT·MPNN·AFP·CNN·LSTM·Transformer)은 수치를 확정하지 않았다.**
표본이 작아 판정할 수 없고, 완주 비용이 아래와 같기 때문이다.
코드는 동작하며 아래 고장 여섯을 모두 우회한다.

## 2022년 코드가 2026년에 깨지는 지점 — 여섯

| # | 고장 | 원인 | 대응 |
|---|---|---|---|
| 1 | `GraphMultisetTransformer` 인자 오류 (GCN·GAT·MPNN) | PyG 2.5+ 시그니처 변경 | `pyg_compat.py` |
| 2 | `fit() got 'use_multiprocessing'` (LSTM) | Keras 3 가 인자 제거 | `keras_compat.py` |
| 3 | 🚨 `'LSTM' has no attribute 'model'` | **사전학습 가중치가 배포본에 없다** | `pretrained_model=None` |
| 4 | `'>=' int vs NoneType` (Transformer) | config 에 `epochs` 없음 → `None` 유입 | 클래스 기본값 조회 |
| 5 | ⭐ 엉뚱한 모델이 로드됨 | **`save_path` 가 상대경로** — 병렬 실행 시 체크포인트 충돌 | 프로세스별 작업 디렉터리 |
| 6 | 맥에서 GPU 미사용 | `"cuda:0" if cuda.is_available() else "cpu"` 하드코딩 | `device_compat.py` |

**③⑤가 조용히 위험하다.** ③ 때문에 **공식 LSTM 수치(0.7423)는 배포 산출물만으로 재현 불가**다
(`pretrained_lstm.h5` 가 pip·GitHub·릴리스 어디에도 없다. 만드는 스크립트만 있다).
⑤ 는 **단독 실행에서는 절대 나타나지 않고 병렬로 돌려야만** 드러난다.

## 비용 — 이 벤치마크의 딥러닝 절반은 GPU 없이 재현할 수 없다

M5 Pro(18코어·48GB), 모델별 8프로세스 병렬, 공식 프로토콜 기준 **표적당**:

```
MLP 6분 · GCN·AFP 33분 · GAT 50분 · CNN 110분 · LSTM 166분
MPNN 14.5시간  (1겹에 2시간 54분)  → 30표적 = 18일
```

원저자는 GPU 로 돌렸다(`benchmark.py` 첫 줄이 `list_physical_devices('GPU')`).
전통 ML 480셀이 1시간인 것과 차원이 다르다.

### ⚠️ 맥 GPU(MPS)는 모델마다 유불리가 갈린다

```
MPNN  5에폭 · 2,201분자   CPU 341.3초  →  MPS  54.9초   (6.2배 빠름)
GCN   3에폭 · 615분자     CPU   5.5초  →  MPS  11.1초   (2배 느림)
```

분자 그래프가 작아 커널 실행 오버헤드가 계산량을 넘기 쉽고, 손익분기가 모델 무게에 따라 갈린다.
`--device mps` 로 켠다. ⚠️ **MPS 수치는 CPU 와 정확히 일치하지 않는다** — 결과 CSV 의 `device` 열을 함께 인용할 것.

## 실행

```bash
./mace-env/bin/python scripts/run_deep.py --models MPNN --device mps
./mace-env/bin/python scripts/run_deep.py --models GCN GAT --targets CHEMBL204_Ki
```

⚠️ **여러 모델을 동시에 돌릴 때는 반드시 서로 다른 작업 디렉터리에서** 실행할 것(고장 ⑤).

## 다음

- [ ] 딥러닝 7종 (GCN·GAT·MPNN·AFP·CNN·LSTM·Transformer)
- [ ] AFP 가 공식보다 크게 좋게 나오는 현상의 원인 규명 (표본 편향인지 PyG 구현 변경인지)
- [ ] LSTM 사전학습 재현 (`Experiments/pretrain_lstm.py`) — 하면 공식과 대조 가능해진다

## 출처

van Tilborg, Alenicheva, Grisoni. *Exposing the Limitations of Molecular Machine
Learning with Activity Cliffs.* J. Chem. Inf. Model. 2022, 62 (23), 5938-5951.
[doi:10.1021/acs.jcim.2c01073](https://doi.org/10.1021/acs.jcim.2c01073)
