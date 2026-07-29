"""전통 ML 16조합 (4모델 x 4표현 x 30표적).

표현 계산을 (표적, 표현)당 한 번만 하고 4모델이 공유한다.
WHIM 은 컨포머 생성이 들어가 느리므로 이 재사용이 4배 차이를 낸다.
공식 결과표는 없으면 scripts/run_baselines.py 가 받아둔 캐시를 쓴다.
"""
import os, time, urllib.request, warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from MoleculeACE import (Data, Descriptors, calc_rmse, calc_cliff_rmse,
                         get_benchmark_config, SVM, RF, GBM, KNN)

OFFICIAL_URL = ("https://raw.githubusercontent.com/molML/MoleculeACE/main/"
                "MoleculeACE/Data/results/MoleculeACE_results.csv")
CACHE = 'data/MoleculeACE_results.csv'
OUT = 'results/ml16_reproduction.csv'

if not os.path.exists(CACHE):
    os.makedirs(os.path.dirname(CACHE) or '.', exist_ok=True)
    print(f"공식 결과표 내려받는 중 -> {CACHE}")
    urllib.request.urlretrieve(OFFICIAL_URL, CACHE)
off = pd.read_csv(CACHE)

targets = sorted(off.dataset.unique())
DESCS = [('ECFP', Descriptors.ECFP), ('MACCS', Descriptors.MACCS),
         ('PHYSCHEM', Descriptors.PHYSCHEM), ('WHIM', Descriptors.WHIM)]
MODELS = [('SVM', SVM), ('RF', RF), ('GBM', GBM), ('KNN', KNN)]

rows = []
t_start = time.time()
for dname, dobj in DESCS:
    for ti, t in enumerate(targets, 1):
        t0 = time.time()
        d = Data(t)
        d(dobj)                      # 표현 계산 1회 — 아래 4모델이 공유
        feat = time.time() - t0
        for mname, cls in MODELS:
            hp = get_benchmark_config(t, cls, dobj)
            m = cls(**hp)
            m.train(d.x_train, d.y_train)
            yh = m.predict(d.x_test)
            r = calc_rmse(d.y_test, yh)
            rc = calc_cliff_rmse(y_test_pred=yh, y_test=d.y_test,
                                 cliff_mols_test=d.cliff_mols_test)
            row = dict(descriptor=dname, algorithm=mname, dataset=t,
                       rmse_mine=r, cliff_mine=rc,
                       n_train=len(d.y_train), n_test=len(d.y_test),
                       n_cliff=int(sum(d.cliff_mols_test)))
            hit = off[(off.dataset == t) & (off.algorithm == mname)
                      & (off.descriptor == dname)]
            if len(hit):
                o = hit.iloc[0]
                row.update(rmse_off=o.rmse, d_rmse=r - o.rmse,
                           cliff_off=o.cliff_rmse, d_cliff=rc - o.cliff_rmse)
            rows.append(row)
        print(f"[{dname:8s} {ti:2d}/30] {t:18s} 표현 {feat:5.1f}s  "
              f"전체 {time.time()-t0:5.1f}s  누적 {(time.time()-t_start)/60:.1f}분",
              flush=True)
        os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)   # 중간 저장

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"\n완료 {len(df)}행  총 {(time.time()-t_start)/60:.1f}분\n")
print("=== 조합별 (재현 / 공식) ===")
g = df.groupby(['descriptor', 'algorithm']).agg(
    rm=('rmse_mine', 'mean'), ro=('rmse_off', 'mean'),
    cm=('cliff_mine', 'mean'), co=('cliff_off', 'mean')).round(4)
print(g.to_string())
