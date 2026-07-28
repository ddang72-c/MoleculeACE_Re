# MoleculeACE_Re

[MoleculeACE](https://github.com/molML/MoleculeACE) 벤치마크의 ECFP 기준선을
최신 파이썬 환경에서 재현한 것. 활동 절벽(activity cliff) 실험을 팀이 공통으로
올려놓을 바닥을 만드는 것이 목적이다.

**이미 나와 있는 숫자를 다시 만드는 게 목적이 아니다.** 공식 결과표(720행)는
패키지에 그대로 들어 있다. 필요한 것은 **새 방법을 돌렸을 때 그 성적을 공식 표와
나란히 놓을 수 있는 파이프라인**이고, 그러려면 내 코드가 공식과 같은 조건에 서
있다는 증명이 먼저 있어야 한다. 이 재현이 그 증명이다.

## 범위

**모델 4종 x 표현 1종(ECFP) x 표적 30개 = 120개 조합.**
표현은 ECFP만 돌렸다. MACCS/그래프/SMILES 계열은 아직 없다.

## 결과

| 모델 | RMSE 재현 | RMSE 공식 | 차 | cliff 재현 | cliff 공식 | 차 | 소수4자리 일치 | 최대 셀오차 |
|---|---:|---:|---:|---:|---:|---:|:--:|---:|
| **SVM** | 0.6712 | 0.6712 | −0.000002 | 0.7511 | 0.7511 | −0.000004 | **30/30 · 29/30** | 0.0001 |
| **GBM** | 0.6953 | 0.6948 | +0.0004 | 0.7836 | 0.7835 | +0.0002 | 0/30 · 0/30 | 0.0519 |
| **RF** | 0.7015 | 0.7000 | +0.0015 | 0.7844 | 0.7857 | −0.0013 | 2/30 · 1/30 | 0.0402 |
| **KNN** | 0.7292 | 0.7290 | +0.0003 | 0.8393 | 0.8388 | +0.0005 | 22/30 · 22/30 | 0.0062 |

표적별 상세: [`results/ecfp_baselines_reproduction.csv`](results/ecfp_baselines_reproduction.csv)

### SVM만 정확히 재현된다 — 그리고 그게 맞다

**트리 앙상블에 시드가 박혀 있지 않다.** `get_benchmark_config`가 RF에
`{'n_estimators': 500}`만 돌려주고 `random_state`가 없으며, 패키지의
`RANDOM_SEED = 42`도 이들에는 적용되지 않는다. 같은 표적을 세 번 돌리면:

```
RF   0.665992   0.658890   0.665117     <- 매번 다름
SVM  0.576209   0.576209   0.576209     <- 결정적
```

**따라서 RF·GBM에는 표적별 일치가 애초에 맞는 검사가 아니다.** 평균이 공식 대비
0.0015 안에 든다는 것이 시드 없이 주장할 수 있는 전부다.

**같이 기억할 것:** 네 모델의 평균은 ~0.001로 맞는데 **개별 모델x표적 셀은 최대
0.05까지 벌어진다.** 집계가 맞는다고 셀이 맞는 게 아니다.

셀 단위로 정확히 재현해야 한다면 `random_state`를 명시하고 그 값을 함께 적어야
한다. 그러지 않으면 공식 표와 셀 대조는 불가능하다.

## 환경 구축 — 실제로 막히는 지점

`pip install MoleculeACE`만으로는 **동작하지 않는다.** 패키지 `__init__.py`
첫 줄이 `from MoleculeACE.models.mpnn import MPNN`이라서, **SVM 하나만 쓰려
해도 딥러닝 스택 전체가 로드된다.**

```bash
uv venv --python 3.11 mace-env
uv pip install --python mace-env/bin/python MoleculeACE
uv pip install --python mace-env/bin/python torch torch_geometric transformers tensorflow
uv pip install --python mace-env/bin/python -U rdkit
```

막혔던 순서:

| # | 오류 | 원인 |
|---|---|---|
| 1 | `No module named 'torch'` | `__init__` → MPNN |
| 2 | `No module named 'torch_geometric'` | 같음 |
| 3 | `No module named 'transformers'` | `benchmark/utils.py` |
| 4 | `No module named 'tensorflow'` | 같음 |
| 5 | `AttributeError: _ARRAY_API not found` | **numpy 2.4 ↔ rdkit 2022.09.5 ABI 충돌** |

**5번이 마지막 고비다.** 패키지가 끌어오는 rdkit이 numpy 1 시절 빌드라서
rdkit 2026.3.4로 올려야 통과한다.

원 패키지 README는 Python 3.8 / TensorFlow 2.9 / PyTorch 1.11(2022년 스택)을
요구하지만 **따르지 않아도 된다.** 위 수치는 Python 3.11, torch 2.13, PyG 2.8,
transformers 5.14, rdkit 2026.3.4에서 나왔다.

## 실행

```bash
./mace-env/bin/python scripts/run_ecfp_baselines.py    # SVM, RF, GBM, KNN
./mace-env/bin/python scripts/run_svm_ecfp_30.py       # SVM만
```

하이퍼파라미터는 `get_benchmark_config`가 제공하므로 **튜닝 단계가 없다.**
SVM은 30표적에 약 20초, 앙상블은 더 걸린다.

## 다음

- [ ] RF·GBM 시드를 고정할지 결정하고 그 선택을 기록
- [ ] 표현 확장 (MACCS, 그래프, SMILES)
- [ ] 표적·모델을 인자로 받도록 일반화
- [ ] 팀원별 담당 모델 분담

## 출처

van Tilborg, Alenicheva, Grisoni. *Exposing the Limitations of Molecular Machine
Learning with Activity Cliffs.* J. Chem. Inf. Model. 2022, 62 (23), 5938–5951.
[doi:10.1021/acs.jcim.2c01073](https://doi.org/10.1021/acs.jcim.2c01073)
