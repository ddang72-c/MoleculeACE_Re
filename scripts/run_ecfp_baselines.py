import warnings, os, time
warnings.filterwarnings("ignore"); os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import pandas as pd
from MoleculeACE import Data, Descriptors, calc_rmse, calc_cliff_rmse, get_benchmark_config, SVM, RF, GBM, KNN

off = pd.read_csv('/tmp/mace.csv')
targets = sorted(off.dataset.unique())
MODELS = [('SVM',SVM),('RF',RF),('GBM',GBM),('KNN',KNN)]
rows=[]
for name,cls in MODELS:
    for i,t in enumerate(targets,1):
        d=Data(t); hp=get_benchmark_config(t, cls, Descriptors.ECFP); d(Descriptors.ECFP)
        m=cls(**hp); m.train(d.x_train,d.y_train); yh=m.predict(d.x_test)
        r=calc_rmse(d.y_test,yh)
        rc=calc_cliff_rmse(y_test_pred=yh,y_test=d.y_test,cliff_mols_test=d.cliff_mols_test)
        o=off[(off.dataset==t)&(off.algorithm==name)&(off.descriptor=='ECFP')].iloc[0]
        rows.append(dict(algorithm=name,dataset=t,rmse_mine=r,rmse_off=o.rmse,d_rmse=r-o.rmse,
                         cliff_mine=rc,cliff_off=o.cliff_rmse,d_cliff=rc-o.cliff_rmse,
                         n_train=len(d.y_train),n_test=len(d.y_test),n_cliff=int(sum(d.cliff_mols_test))))
    print(f"{name} done", flush=True)
df=pd.DataFrame(rows); df.to_csv('ecfp_baselines_reproduction.csv', index=False)
print("\n=== per-model summary ===")
g=df.groupby('algorithm').agg(rmse_mine=('rmse_mine','mean'),rmse_off=('rmse_off','mean'),
                              cliff_mine=('cliff_mine','mean'),cliff_off=('cliff_off','mean'))
g['d_rmse']=g.rmse_mine-g.rmse_off; g['d_cliff']=g.cliff_mine-g.cliff_off
print(g.round(6).to_string())
print("\nagreement to 4 decimals")
for n in df.algorithm.unique():
    s=df[df.algorithm==n]
    print(f"  {n:4s} RMSE {(s.d_rmse.abs()<5e-5).sum()}/30   cliff {(s.d_cliff.abs()<5e-5).sum()}/30   max err {max(s.d_rmse.abs().max(),s.d_cliff.abs().max()):.6f}")
