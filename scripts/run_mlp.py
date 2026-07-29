"""MLP + ECFP 를 30 표적에서 재현한다.

전통 ML 16조합(run_ml16.py)과 달리 MLP 는 공식 벤치마크에서 다른 절차로 학습된다.
Experiments/benchmark.py 를 보면 딥러닝 계열은 단일 학습이 아니다.

    hyperparameters.pop("epochs")
    cross_validate(algo, data, n_folds=5, early_stopping=10, seed=RANDOM_SEED)
    rmse = sum(rmse) / len(rmse)

즉 5겹 교차검증 평균이고 조기종료 patience 는 10 이다. 이 절차를 맞추지 않으면
같은 모델이라도 다른 실험이 된다. ECFP 는 증강 대상이 아니므로 증강은 하지 않는다
(공식은 SMILES·TOKENS 표현에만 10배 증강을 건다).
"""
import argparse
import os
import time
import urllib.request
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
from MoleculeACE import Data, Descriptors, MLP, get_benchmark_config
from MoleculeACE.benchmark.const import RANDOM_SEED
from MoleculeACE.benchmark.utils import cross_validate

OFFICIAL_CSV = (
    "https://raw.githubusercontent.com/molML/MoleculeACE/main/"
    "MoleculeACE/Data/results/MoleculeACE_results.csv"
)

ap = argparse.ArgumentParser(description="MLP + ECFP 재현 (공식 프로토콜)")
ap.add_argument("--targets", nargs="+", default=None, help="표적 선택 (기본: 30개 전부)")
ap.add_argument("--folds", type=int, default=5, help="교차검증 겹 수. 공식은 5")
ap.add_argument("--out", default="results/mlp_ecfp_reproduction.csv")
ap.add_argument("--cache", default="data/MoleculeACE_results.csv")
args = ap.parse_args()

if args.folds != 5:
    warnings.warn(f"겹 수가 {args.folds} 다. 공식은 5 이므로 결과를 공식표와 대조할 수 없다.")

if not os.path.exists(args.cache):
    os.makedirs(os.path.dirname(args.cache) or ".", exist_ok=True)
    urllib.request.urlretrieve(OFFICIAL_CSV, args.cache)
official = pd.read_csv(args.cache)
targets = args.targets or sorted(official.dataset.unique())

rows = []
t0 = time.time()
for i, target in enumerate(targets, 1):
    started = time.time()
    data = Data(target)
    data(Descriptors.ECFP)

    hp = get_benchmark_config(target, MLP, Descriptors.ECFP)
    hp.pop("epochs", None)  # 조기종료가 학습 길이를 정한다

    fold_rmse, fold_cliff = cross_validate(
        MLP, data,
        n_folds=args.folds, early_stopping=10, seed=RANDOM_SEED, save_path=None,
        **hp,
    )
    row = {
        "algorithm": "MLP",
        "descriptor": "ECFP",
        "dataset": target,
        "rmse_mine": sum(fold_rmse) / len(fold_rmse),
        "cliff_mine": sum(fold_cliff) / len(fold_cliff),
        "rmse_folds": ";".join(f"{v:.4f}" for v in fold_rmse),
        "n_folds": args.folds,
    }
    hit = official[
        (official.dataset == target)
        & (official.algorithm == "MLP")
        & (official.descriptor == "ECFP")
    ]
    if len(hit):
        row["rmse_off"] = hit.iloc[0].rmse
        row["cliff_off"] = hit.iloc[0].cliff_rmse

    rows.append(row)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(
        f"[{i:2d}/{len(targets)}] {target:18s} "
        f"{row['rmse_mine']:.4f} / {row['cliff_mine']:.4f}  "
        f"({time.time() - started:.0f}s, 누적 {(time.time() - t0) / 60:.1f}분)",
        flush=True,
    )

df = pd.DataFrame(rows)
print(f"\n완료 {len(df)}표적 · {(time.time() - t0) / 60:.1f}분\n")
print(
    df[["rmse_mine", "rmse_off", "cliff_mine", "cliff_off"]]
    .mean()
    .round(4)
    .to_string()
)
