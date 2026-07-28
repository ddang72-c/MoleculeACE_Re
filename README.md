# MoleculeACE_Re

[MoleculeACE](https://github.com/molML/MoleculeACE) 벤치마크 기준선을 돌리는 공통 코드.
새 방법을 만들었을 때 그 성적을 공식 결과표와 나란히 놓으려면, 먼저 내 코드가 공식과
같은 조건에서 돌아간다는 것을 보여야 한다.

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

- 패키지 `__init__.py` 가 MPNN 을 먼저 불러서 **SVM 하나만 쓰어도 torch/PyG/
  transformers/tensorflow 가 전부 필요하다.**
- **마지막 `-U rdkit` 을 빼면 `AttributeError: _ARRAY_API not found` 로 죽는다.**
  패키지가 끌어오는 rdkit 2022.09.5 가 numpy 2 와 충돌한다.

원 패키지 README 가 요구하는 Python 3.8 / TF 2.9 / PyTorch 1.11 은 **따를 필요 없다.**
데이터셋은 `pip install` 할 때 패키지 안에 같이 깔린다(7MB, 따로 받을 필요 없음).

## 실행

```bash
./mace-env/bin/python scripts/run_baselines.py
```

결과가 `results/reproduction.csv` 로 나온다. 공식 결과표는 없으면 자동으로 받아온다.

조합을 골라서 돌리려면:

```bash
--models SVM RF            # 모델 고르기 (SVM RF GBM KNN MLP)
--descriptor MACCS        # 표현 고르기 (ECFP MACCS PHYSCHEM WHIM)
--targets CHEMBL204_Ki    # 표적 고르기
--seed 42                 # 시드 고정
--out results/my.csv      # 출력 경로
```

예: `./mace-env/bin/python scripts/run_baselines.py --models GBM --seed 42`

하이퍼파라미터는 `get_benchmark_config` 가 주므로 튜닝할 것이 없다.
SVM 은 30표적에 약 20초, 앙상블은 더 걸린다.

## ⚠️ 시드를 반드시 고정할 것

**RF·GBM 은 시드가 없으면 돌릴 때마다 숫자가 바뀜다.**
`get_benchmark_config` 가 `random_state` 를 주지 않고 패키지의
`RANDOM_SEED = 42` 도 적용되지 않는다.

```
같은 표적을 RF 로 3번:   0.665992   0.658890   0.665117
같은 표적을 SVM 으로 3번:  0.576209   0.576209   0.576209
```

**결과를 공유할 때는 `--seed` 를 쓰고 그 값을 같이 적어주세요.**
그러지 않으면 서로의 숫자를 비교할 수 없습니다. SVM·KNN 은 결정적이라 상관없습니다.

## 현재 돌린 범위

**모델 4종 x 표현 1종(ECFP) x 표적 30개 = 120개** (시드 없이 실행)

| 모델 | RMSE | 공식 | cliff | 공식 | 셀 단위 일치 |
|---|---:|---:|---:|---:|:--:|
| SVM | 0.6712 | 0.6712 | 0.7511 | 0.7511 | 30/30 |
| GBM | 0.6953 | 0.6948 | 0.7836 | 0.7835 | 0/30 |
| RF | 0.7015 | 0.7000 | 0.7844 | 0.7857 | 2/30 |
| KNN | 0.7292 | 0.7290 | 0.8393 | 0.8388 | 22/30 |

평균은 네 모델 다 공식 대비 0.0015 안에 든다. 셀 단위로 갈리는 것은 위 시드 문제다.
표적별 상세는 [`results/ecfp_baselines_reproduction.csv`](results/ecfp_baselines_reproduction.csv).

**아직 안 돌린 것:** MACCS·PHYSCHEM·WHIM 표현, 그래프·SMILES 계열 모델.

## 출처

van Tilborg, Alenicheva, Grisoni. *Exposing the Limitations of Molecular Machine
Learning with Activity Cliffs.* J. Chem. Inf. Model. 2022, 62 (23), 5938–5951.
[doi:10.1021/acs.jcim.2c01073](https://doi.org/10.1021/acs.jcim.2c01073)
