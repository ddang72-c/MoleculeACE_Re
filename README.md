# MoleculeACE_Re

[MoleculeACE](https://github.com/molML/MoleculeACE) 벤치마크 기준선을 돌리는 공통 코드.
새 방법을 만들었을 때 그 성적을 공식 결과표와 나란히 놓으려면, 먼저 내 코드가 공식과
같은 조건에서 돌아간다는 것을 보여야 한다.

## 범위

**전통 ML 16조합 = 모델 4종(SVM·RF·GBM·KNN) x 표현 4종(ECFP·MACCS·PHYSCHEM·WHIM) x 표적 30개 = 480개.**
공식 24접근 중 딥러닝 8종(그래프·SMILES·TOKENS)은 아직 없다.

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

### ② PHYSCHEM — `NumHAcceptors` 정의가 rdkit 버전 간 바뀜었다

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
바뀜어서 같은 시드로도 다른 좌표가 나온다.** 같은 분자의 WHIM 114개 중 4개가 달라진다
(최대차 0.001). 표적 전체로 누적되면 셀 단위 불일치가 된다.

> **⇒ 정리:** numpy 2 때문에 rdkit을 올려야 하는데, **그 업그레이드가 PHYSCHEM과 WHIM 값을
> 바꿈다.** ECFP·MACCS는 비트 정의가 고정이라 영향이 없다.
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
--models SVM RF           # 모델 고르기 (SVM RF GBM KNN MLP)
--descriptor MACCS        # 표현 고르기 (ECFP MACCS PHYSCHEM WHIM)
--targets CHEMBL204_Ki    # 표적 고르기
--seed 42                 # 시드 고정 (RF/GBM/MLP 에만 적용)
--out results/my.csv      # 출력 경로
```

하이퍼파라미터는 `get_benchmark_config`가 주므로 튜닝할 것이 없다.
SVM+ECFP는 30표적에 약 20초, WHIM은 컨포머 생성 때문에 표적당 1~8분 걸린다.

## 다음

- [ ] 딥러닝 8종(그래프·SMILES·TOKENS) 추가
- [ ] 팀원별 담당 모델 분담

## 출처

van Tilborg, Alenicheva, Grisoni. *Exposing the Limitations of Molecular Machine
Learning with Activity Cliffs.* J. Chem. Inf. Model. 2022, 62 (23), 5938-5951.
[doi:10.1021/acs.jcim.2c01073](https://doi.org/10.1021/acs.jcim.2c01073)
