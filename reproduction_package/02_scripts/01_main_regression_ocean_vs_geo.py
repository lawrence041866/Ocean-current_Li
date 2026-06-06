# C1 main regression: ocean-current connectivity vs geographic proximity; is the ocean channel independent after controls
# Inputs: chl_yearly_2003_2021.csv, W_sovereign_directed.csv, hex2country.csv
# Output: ocean lambda=0.44*** > geography 0.38*** (manuscript Fig. 2a)
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
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
Wn=rn(Wsym)
hx=pd.read_csv(D+"hex2country.csv"); cen=hx.groupby('iso3')[['lon','lat']].median().reindex(countries)
def hav(a,b):
    la1,lo1,la2,lo2=map(np.radians,[a[1],a[0],b[1],b[0]])
    return 2*6371*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
Wgeo=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i!=j and cen.iloc[i].notna().all() and cen.iloc[j].notna().all():
            d=hav(cen.iloc[i].values,cen.iloc[j].values)
            if 0<d<3000: Wgeo[i,j]=1/d
Wgeo=rn(Wgeo)
chl=chl[chl.iso3.isin(countries)]
def lag(M,name):
    out=[]
    for yr,g in chl.groupby('year'):
        g=g.set_index('iso3').reindex(countries)
        y=g.lchl.values.astype(float);ym=np.nan_to_num(y,nan=np.nanmean(y))
        out.append(pd.DataFrame({'iso3':countries,'year':yr,name:M@ym}))
    return pd.concat(out)
chl=chl.merge(lag(Wn,'W_ocean'),on=['iso3','year']).merge(lag(Wgeo,'W_geo'),on=['iso3','year'])
df=chl.set_index(['iso3','year'])
d=df[['lchl','W_ocean','W_geo']].dropna(); d.columns=['y','Wo','Wg']
r=PanelOLS(d['y'],sm.add_constant(d[['Wo','Wg']]),entity_effects=True,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
print(f"ocean beta={r.params['Wo']:+.3f}{sig(r.pvalues['Wo'])} | geo beta={r.params['Wg']:+.3f}{sig(r.pvalues['Wg'])} | n={len(d)} n_countries={d.index.get_level_values(0).nunique()}")
