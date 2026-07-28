"""MoleculeACE 기준선 재현 — 표적·모델·표현을 인자로 받는다.

사용 예:
    python scripts/run_baselines.py                          # 전체 (4모델 x ECFP x 30표적)
    python scripts/run_baselines.py --models SVM             # SVM만
    python scripts/run_baselines.py --models SVM RF          # 두 모델
    python scripts/run_baselines.py --targets CHEMBL204_Ki   # 한 표적만
    python scripts/run_baselines.py --descriptor MACCS       # 표현 바꾸기
    python scripts/run_baselines.py --models RF --seed 42    # 시드 고정
"""
import argparse, os, time, urllib.request, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import pandas as pd
from MoleculeACE import (Data, Descriptors, calc_rmse, calc_cliff_rmse,
                         get_benchmark_config, SVM, RF, GBM, KNN, MLP)

OFFICIAL_URL = ("https://raw.githubusercontent.com/molML/MoleculeACE/main/"
                "MoleculeACE/Data/results/MoleculeACE_results.csv")
MODELS = {'SVM': SVM, 'RF': RF, 'GBM': GBM, 'KNN': KNN, 'MLP': MLP}
# random_state 를 받는 모델만. SVM/KNN 은 애초에 결정적이라 시드가 필요 없다.
SEEDABLE = {'RF', 'GBM', 'MLP'}
DESCRIPTORS = {'ECFP': Descriptors.ECFP, 'MACCS': Descriptors.MACCS,
               'PHYSCHEM': Descriptors.PHYSCHEM, 'WHIM': Descriptors.WHIM}


def load_official(cache):
    """공식 결과표를 받아 캐시한다. 없으면 내려받는다."""
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache) or '.', exist_ok=True)
        print(f"공식 결과표 내려받는 중 -> {cache}")
        urllib.request.urlretrieve(OFFICIAL_URL, cache)
    return pd.read_csv(cache)


def main():
    p = argparse.ArgumentParser(description="MoleculeACE 기준선 재현")
    p.add_argument('--models', nargs='+', default=['SVM', 'RF', 'GBM', 'KNN'],
                   choices=list(MODELS), help="돌릴 모델 (기본: 4종 전부)")
    p.add_argument('--descriptor', default='ECFP', choices=list(DESCRIPTORS),
                   help="분자 표현 (기본: ECFP)")
    p.add_argument('--targets', nargs='+', default=None,
                   help="표적 ID (기본: 30개 전부)")
    p.add_argument('--seed', type=int, default=None,
                   help="RF/GBM/MLP 시드 고정. 주지 않으면 매 실행마다 결과가 "
                        "달라진다. SVM/KNN 은 결정적이라 영향 없음")
    p.add_argument('--cache', default='data/MoleculeACE_results.csv',
                   help="공식 결과표 캐시 경로")
    p.add_argument('--out', default='results/reproduction.csv', help="출력 경로")
    a = p.parse_args()

    off = load_official(a.cache)
    targets = a.targets or sorted(off.dataset.unique())
    desc = DESCRIPTORS[a.descriptor]

    print(f"모델 {a.models} x 표현 {a.descriptor} x 표적 {len(targets)}개 "
          f"= {len(a.models) * len(targets)}개 조합"
          + (f" (시드 {a.seed})" if a.seed is not None else " (시드 없음)"))

    rows = []
    for name in a.models:
        cls = MODELS[name]
        for t in targets:
            t0 = time.time()
            d = Data(t)
            hp = get_benchmark_config(t, cls, desc)
            if a.seed is not None and name in SEEDABLE:
                hp = {**hp, 'random_state': a.seed}
            d(desc)
            m = cls(**hp)
            m.train(d.x_train, d.y_train)
            yh = m.predict(d.x_test)
            r = calc_rmse(d.y_test, yh)
            rc = calc_cliff_rmse(y_test_pred=yh, y_test=d.y_test,
                                 cliff_mols_test=d.cliff_mols_test)
            row = dict(algorithm=name, descriptor=a.descriptor, dataset=t,
                       rmse_mine=r, cliff_mine=rc,
                       seed=a.seed if name in SEEDABLE else None,
                       n_train=len(d.y_train), n_test=len(d.y_test),
                       n_cliff=int(sum(d.cliff_mols_test)),
                       sec=round(time.time() - t0, 1))
            hit = off[(off.dataset == t) & (off.algorithm == name)
                      & (off.descriptor == a.descriptor)]
            if len(hit):
                o = hit.iloc[0]
                row.update(rmse_off=o.rmse, d_rmse=r - o.rmse,
                           cliff_off=o.cliff_rmse, d_cliff=rc - o.cliff_rmse)
            rows.append(row)
            print(f"  {name:4s} {t:18s} {r:.4f} / {rc:.4f}"
                  f"  ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\n저장: {a.out}  ({len(df)}행)")

    if 'd_rmse' in df:
        print("\n=== 공식 대비 ===")
        for name, s in df.groupby('algorithm'):
            s = s.dropna(subset=['d_rmse'])
            if not len(s):
                continue
            n = len(s)
            print(f"  {name:4s} RMSE {s.rmse_mine.mean():.4f} vs {s.rmse_off.mean():.4f}"
                  f"  cliff {s.cliff_mine.mean():.4f} vs {s.cliff_off.mean():.4f}"
                  f"   소수4자리 {(s.d_rmse.abs()<5e-5).sum()}/{n}"
                  f"   최대오차 {max(s.d_rmse.abs().max(), s.d_cliff.abs().max()):.6f}")


if __name__ == '__main__':
    main()
