# MoleculeACE_Re

[MoleculeACE](https://github.com/molML/MoleculeACE) 벤치마크의 기준선을 재현하는 공통 코드다.

새 방법을 만들었을 때 그 성적을 공식 결과표와 나란히 놓으려면, 먼저 우리 코드가 공식과
같은 조건에서 돌아간다는 것을 보여야 한다. 이 저장소가 그 확인이다.

## 범위

공식 24개 접근 중 17개를 재현했다.

| 계열 | 구성 | 개수 | 상태 |
|---|---|--:|---|
| 전통 ML | 모델 4종(SVM·RF·GBM·KNN) × 표현 4종(ECFP·MACCS·PHYSCHEM·WHIM) | 16 | 완료 · 480셀 |
| MLP + ECFP | 신경망 기준선 | 1 | 완료 · 30셀 |

17 × 30표적 = 510셀. 나머지 7종(GCN·GAT·MPNN·AFP·CNN·LSTM·Transformer)은 다루지 않는다.

## 결과

### 전통 ML 16조합

| 표현 | 모델 | RMSE 재현 | 공식 | 차 | cliff 재현 | 공식 | 차 | 셀 일치 |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| ECFP | SVM | 0.6712 | 0.6712 | -0.0000 | 0.7511 | 0.7511 | -0.0000 | 30/30 |
| ECFP | GBM | 0.6932 | 0.6948 | -0.0016 | 0.7870 | 0.7835 | +0.0036 | 0/30 |
| ECFP | RF | 0.7009 | 0.7000 | +0.0009 | 0.7847 | 0.7857 | -0.0010 | 0/30 |
| ECFP | KNN | 0.7292 | 0.7290 | +0.0003 | 0.8393 | 0.8388 | +0.0005 | 22/30 |
| MACCS | GBM | 0.7490 | 0.7495 | -0.0005 | 0.8462 | 0.8416 | +0.0046 | 0/30 |
| MACCS | SVM | 0.7520 | 0.7520 | -0.0000 | 0.8499 | 0.8499 | +0.0000 | 30/30 |
| MACCS | RF | 0.7561 | 0.7557 | +0.0004 | 0.8471 | 0.8472 | -0.0001 | 0/30 |
| MACCS | KNN | 0.8110 | 0.8110 | +0.0000 | 0.8999 | 0.9001 | -0.0002 | 22/30 |
| PHYSCHEM | RF | 0.8848 | 0.8807 | +0.0041 | 0.9350 | 0.9302 | +0.0049 | 0/30 |
| PHYSCHEM | GBM | 0.8935 | 0.8913 | +0.0022 | 0.9460 | 0.9464 | -0.0005 | 0/30 |
| PHYSCHEM | KNN | 0.9176 | 0.9177 | -0.0001 | 0.9479 | 0.9433 | +0.0045 | 0/30 |
| PHYSCHEM | SVM | 0.9348 | 0.9373 | -0.0025 | 0.9646 | 0.9636 | +0.0010 | 0/30 |
| WHIM | RF | 0.9771 | 0.9763 | +0.0008 | 1.0187 | 1.0144 | +0.0043 | 0/30 |
| WHIM | GBM | 0.9908 | 0.9946 | -0.0038 | 1.0396 | 1.0294 | +0.0102 | 0/30 |
| WHIM | SVM | 1.0044 | 1.0083 | -0.0039 | 1.0578 | 1.0516 | +0.0062 | 0/30 |
| WHIM | KNN | 1.0226 | 1.0162 | +0.0064 | 1.0574 | 1.0446 | +0.0128 | 1/30 |

평균 차이는 16조합 전부 0.0064 이내다. 표적별 상세는 [`results/ml16_reproduction.csv`](results/ml16_reproduction.csv).

### MLP + ECFP

| 모델 | RMSE 재현 | 공식 | 차 | cliff 재현 | 공식 | 절벽격차 재현 | 공식 |
|---|--:|--:|--:|--:|--:|--:|--:|
| MLP + ECFP | 0.7352 | 0.7632 | -0.0280 | 0.8090 | 0.8333 | 0.0738 | 0.0701 |

표적별 상세는 [`results/mlp_ecfp_reproduction.csv`](results/mlp_ecfp_reproduction.csv).
전체 평균뿐 아니라 절벽 격차(0.0738 대 0.0701)도 재현된다.

## 설치

```bash
git clone https://github.com/ddang72-c/MoleculeACE_Re
cd MoleculeACE_Re

uv venv --python 3.11 mace-env
uv pip install --python mace-env/bin/python MoleculeACE
uv pip install --python mace-env/bin/python torch torch_geometric transformers tensorflow
uv pip install --python mace-env/bin/python -U rdkit
```

네 줄을 순서대로 실행해야 한다.

- 패키지 `__init__.py` 가 MPNN 을 먼저 불러서, SVM 하나만 쓰더라도
  torch · PyG · transformers · tensorflow 가 모두 필요하다.
- 마지막 `-U rdkit` 을 빼면 `AttributeError: _ARRAY_API not found` 로 죽는다.
  패키지가 끌어오는 rdkit 2022.09.5 가 numpy 2 와 충돌한다.

원 패키지 README 가 요구하는 Python 3.8 / TF 2.9 / PyTorch 1.11 은 따를 필요 없다.
위 결과는 Python 3.11 · torch 2.13 · PyG 2.8 · transformers 5.14 · rdkit 2026.3.4 에서 나왔다.

데이터셋은 `pip install` 할 때 패키지 안에 함께 설치된다(7MB). 따로 받지 않아도 된다.

## 실행

```bash
# 전통 ML 16조합 전부 (WHIM 컨포머 생성 때문에 1시간 이상)
./mace-env/bin/python scripts/run_ml16.py

# 조합을 골라서
./mace-env/bin/python scripts/run_baselines.py --models SVM RF --descriptor MACCS

# MLP + ECFP (5겹 교차검증이라 표적당 몇 분)
./mace-env/bin/python scripts/run_mlp.py
```

`run_baselines.py` 옵션

```
--models SVM RF            모델 (SVM RF GBM KNN)
--descriptor MACCS         표현 (ECFP MACCS PHYSCHEM WHIM)
--targets CHEMBL204_Ki     표적
--seed 42                  시드 고정 (RF·GBM 에만 적용)
--out results/my.csv       출력 경로
```

하이퍼파라미터는 `get_benchmark_config` 가 제공하므로 따로 튜닝할 것이 없다.
SVM+ECFP 는 30표적에 약 20초, WHIM 은 컨포머 생성 때문에 표적당 1~8분 걸린다.

## 주의사항

### 셀 단위로는 조합마다 결과가 갈린다

`ECFP+SVM` 과 `MACCS+SVM` 만 30/30 정확히 맞고 나머지는 맞지 않는다. 원인은 셋이다.

**1. 트리 앙상블에 시드가 없다 (RF · GBM)**

`get_benchmark_config` 가 `random_state` 를 주지 않고, 패키지의 `RANDOM_SEED = 42` 도
적용되지 않는다. 같은 표적을 RF 로 세 번 돌리면 `0.665992 / 0.658890 / 0.665117` 이 나온다.
`--seed` 로 고정할 수 있다.

**2. PHYSCHEM 의 `NumHAcceptors` 정의가 rdkit 버전 간 달라졌다**

PHYSCHEM 은 2D 기술자 11개다. 그중 10개는 rdkit 2022.09.5 와 2026.03.4 에서 값이 같고
`NumHAcceptors` 하나만 다르다.

```
FC(F)(F)c1cccc(-c2nnc3ccc(NC4CCCCC4)cn23)c1
    rdkit 2022.09.5 → NumHAcceptors = 4
    rdkit 2026.03.4 → NumHAcceptors = 3
```

SVM 은 결정적인데도 PHYSCHEM 에서 0/30 인 이유가 이것이다. 모델이 아니라 입력이 달라진다.
HBA SMARTS 가 `nH0` 에서 `nH0X2` 로 바뀐 결과다(RDKit 이슈 #8997, 중성 방향족 질소를
HBA 로 세던 것을 고친 것).

**3. WHIM 은 시드를 고정해도 컨포머가 달라진다**

`EmbedMolecule(..., randomSeed=42)` 로 시드가 박혀 있는데도 같은 분자의 WHIM 114개 중
4개가 달라진다(최대차 0.001). 컨포머 생성·최적화 경로가 버전 간 바뀐 탓이며,
최종 WHIM 값만 비교해서는 `EmbedMolecule` · `MMFFOptimizeMolecule` · `CalcWHIM` 중
어느 단계인지 가릴 수 없다.

정리하면, numpy 2 때문에 rdkit 을 올려야 하는데 그 업그레이드가 PHYSCHEM 과 WHIM 값을
바꾼다. ECFP·MACCS 는 비트 정의가 고정이라 영향이 없다.

> 셀 단위 대조가 필요하면 ECFP·MACCS + SVM 으로 하고,
> PHYSCHEM·WHIM 은 평균으로 비교할 것.

### MLP 는 학습 절차가 다르다

공식 벤치마크에서 딥러닝 계열은 단일 학습이 아니다. `Experiments/benchmark.py` 를 보면

```python
hyperparameters.pop("epochs")
cross_validate(algo, data, n_folds=5, early_stopping=10, seed=RANDOM_SEED)
rmse = sum(rmse) / len(rmse)
```

5겹 교차검증 평균이고 조기종료 patience 는 10 이다. 이 절차를 맞추지 않으면 같은 모델이라도
다른 실험이 된다. `run_mlp.py` 가 이 절차를 그대로 따른다.

전통 ML 16조합은 공식도 단일 학습이므로 영향이 없다.

## 파일

```
scripts/
  run_ml16.py        전통 ML 16조합 전부
  run_baselines.py   조합을 인자로 선택
  run_mlp.py         MLP + ECFP (5겹 교차검증)
results/
  ml16_reproduction.csv         16조합 × 30표적
  mlp_ecfp_reproduction.csv     MLP × 30표적
```

## 출처

van Tilborg, Alenicheva, Grisoni. *Exposing the Limitations of Molecular Machine
Learning with Activity Cliffs.* J. Chem. Inf. Model. 2022, 62(23), 5938-5951.
[doi:10.1021/acs.jcim.2c01073](https://doi.org/10.1021/acs.jcim.2c01073)

원논문에는 정정문이 있다. 학습·테스트 분할의 소프트웨어 버그로 일부 활동절벽 쌍의 라벨이
잘못 붙어 있었고, 저자들이 정정 데이터로 재학습해 결과를 갱신했다. 위 「공식」 값은
정정 후 결과표([`MoleculeACE_results.csv`](https://github.com/molML/MoleculeACE/blob/main/MoleculeACE/Data/results/MoleculeACE_results.csv))
기준이다.
