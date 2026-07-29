"""딥러닝 8종 x 30표적 = 240개.

교수님이 요청한 「ECFP/fingerprint + GNN 기반 베이스 코드」 중 GNN·서열 쪽.

⚠️ 상류 버그 우회 둘
1. train() 을 early_stopping_patience 없이 부르면 patience 가 루프 중 int 가
   되면서 None 과 비교돼 TypeError 가 난다. config 의 epochs 를 그대로 넘겨
   사실상 early stopping 을 끔다.
2. GCN·GAT·MPNN 은 pyg_compat.patch() 없이는 임포트도 되지 않는다.

⚠️ GNN 수치는 호환 계층 때문에 공식 표와 셀 단위로 대조할 수 없다.
"""
import os, time, urllib.request, warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyg_compat import patch
patch()
import pandas as pd
from MoleculeACE import (Data, Descriptors, calc_rmse, calc_cliff_rmse,
                         get_benchmark_config, MLP, GCN, GAT, MPNN, AFP,
                         CNN, LSTM, Transformer)

OUT = 'results/deep_reproduction.csv'
CACHE = 'data/MoleculeACE_results.csv'
OFFICIAL_URL = ('https://raw.githubusercontent.com/molML/MoleculeACE/main/'
                'MoleculeACE/Data/results/MoleculeACE_results.csv')
if not os.path.exists(CACHE):
    os.makedirs(os.path.dirname(CACHE) or '.', exist_ok=True)
    print(f"공식 결과표 내려받는 중 -> {CACHE}")
    urllib.request.urlretrieve(OFFICIAL_URL, CACHE)
off = pd.read_csv(CACHE)
targets = sorted(off.dataset.unique())

# (이름, 클래스, 표현명, 표현객체)
COMBOS = [
    ('MLP',         MLP,         'ECFP',   Descriptors.ECFP),
    ('GCN',         GCN,         'GRAPH',  Descriptors.GRAPH),
    ('GAT',         GAT,         'GRAPH',  Descriptors.GRAPH),
    ('MPNN',        MPNN,        'GRAPH',  Descriptors.GRAPH),
    ('AFP',         AFP,         'GRAPH',  Descriptors.GRAPH),
    ('CNN',         CNN,         'SMILES', Descriptors.SMILES),
    ('LSTM',        LSTM,        'SMILES', Descriptors.SMILES),
    ('Transformer', Transformer, 'TOKENS', Descriptors.TOKENS),
]

rows, t0 = [], time.time()
for name, cls, dname, dobj in COMBOS:
    for ti, t in enumerate(targets, 1):
        s = time.time()
        try:
            d = Data(t)
            hp = get_benchmark_config(t, cls, dobj)
            d(dobj)
            ep = hp.get('epochs')
            m = cls(**hp)
            try:
                m.train(d.x_train, d.y_train,
                        early_stopping_patience=ep, epochs=ep, print_every_n=10**9)
            except TypeError:
                m.train(d.x_train, d.y_train)      # 시그니처가 다른 모델 대비
            yh = m.predict(d.x_test)
            r = calc_rmse(d.y_test, yh)
            rc = calc_cliff_rmse(y_test_pred=yh, y_test=d.y_test,
                                 cliff_mols_test=d.cliff_mols_test)
            row = dict(algorithm=name, descriptor=dname, dataset=t,
                       rmse_mine=r, cliff_mine=rc, error='')
        except Exception as e:
            row = dict(algorithm=name, descriptor=dname, dataset=t,
                       rmse_mine=None, cliff_mine=None,
                       error=type(e).__name__ + ': ' + str(e)[:90])
        hit = off[(off.dataset == t) & (off.algorithm == name)
                  & (off.descriptor == dname)]
        if len(hit):
            o = hit.iloc[0]
            row.update(rmse_off=o.rmse, cliff_off=o.cliff_rmse)
        rows.append(row)
        os.makedirs(os.path.dirname(OUT) or '.', exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
        msg = row['error'] or f"{row['rmse_mine']:.4f}/{row['cliff_mine']:.4f}"
        print(f"[{name:11s} {ti:2d}/30] {t:18s} {msg}  ({time.time()-s:.0f}s "
              f"누적 {(time.time()-t0)/60:.1f}분)", flush=True)

df = pd.DataFrame(rows)
print(f"\n완료 {len(df)}행 · 실패 {df.error.astype(bool).sum()}건 · {(time.time()-t0)/60:.1f}분\n")
ok = df[df.rmse_mine.notna()]
g = ok.groupby('algorithm').agg(rm=('rmse_mine', 'mean'), ro=('rmse_off', 'mean'),
                                cm=('cliff_mine', 'mean'), co=('cliff_off', 'mean'),
                                n=('rmse_mine', 'size'))
print(g.round(4).to_string())
