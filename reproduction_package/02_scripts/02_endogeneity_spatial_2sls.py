# Spatial 2SLS (Kelejian-Prucha): instruments = first/second-order spatial lags of previous-period pollution
# Output: 2SLS lambda=0.98***, first-stage F=96 (manuscript Fig. 2b)
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from linearmodels.iv import IV2SLS
import statsmodels.api as sm
sig=lambda p:'***' if p<.01 else '**' if p<.05 else '*' if p<.1 else ''
D="01_data/"
chl=pd.read_csv(D+"chl_yearly_2003_2021.csv"); chl['year']=chl.year.astype(int)
chl=chl.groupby(['iso3','year']).apply(lambda g:np.average(g.chlor_a_mean,weights=g.pixel_count)).reset_index(name='chl')
chl['lchl']=np.log(chl.chl)
W=pd.read_csv(D+"W_sovereign_directed.csv")
countries=sorted(set(chl.iso3)&(set(W.src)|set(W.dst)))
idx={c:i for i,c in enumerate(countries)};N=len(countries)
Wsym=np.zeros((N,N))
for r in W.itertuples():
    if r.src in idx and r.dst in idx and r.src!=r.dst:
        Wsym[idx[r.src],idx[r.dst]]+=r.n_events; Wsym[idx[r.dst],idx[r.src]]+=r.n_events
rn=lambda M:(M/np.where(M.sum(1,keepdims=True)==0,1,M.sum(1,keepdims=True)))
Wn=rn(Wsym); W2=Wn@Wn
chl=chl[chl.iso3.isin(countries)]
piv=chl.pivot(index='iso3',columns='year',values='lchl').reindex(countries)
yrs=sorted(chl.year.unique()); rows=[]
for t in range(1,len(yrs)):
    yr,prev=yrs[t],yrs[t-1]
    y=piv[yr].values.astype(float); yp=piv[prev].values.astype(float)
    ym=np.nan_to_num(y,nan=np.nanmean(y)); ypm=np.nan_to_num(yp,nan=np.nanmean(yp))
    Wy=Wn@ym; Wy_lag=Wn@ypm; W2y_lag=W2@ypm
    for i,c in enumerate(countries):
        if not np.isnan(y[i]): rows.append([c,yr,y[i],Wy[i],Wy_lag[i],W2y_lag[i]])
d=pd.DataFrame(rows,columns=['iso3','year','y','Wy','Wy_lag','W2y_lag'])
for col in ['y','Wy','Wy_lag','W2y_lag']:
    d[col]=d.groupby('iso3')[col].transform(lambda x:x-x.mean())
    d[col]=d[col]-d.groupby('year')[col].transform('mean')
ols=sm.OLS(d.y,d[['Wy']]).fit(cov_type='cluster',cov_kwds={'groups':d.iso3})
iv=IV2SLS(d.y,sm.add_constant(pd.DataFrame(index=d.index)),d[['Wy']],d[['Wy_lag','W2y_lag']]).fit(cov_type='clustered',clusters=d.iso3)
print(f"OLS λ={ols.params['Wy']:+.3f}{sig(ols.pvalues['Wy'])} | 2SLS λ={iv.params['Wy']:+.3f}{sig(iv.pvalues['Wy'])} F={iv.first_stage.diagnostics['f.stat'].values[0]:.0f}")
