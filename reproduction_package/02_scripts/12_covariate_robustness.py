# Covariate robustness (M5): does the ocean-connectivity coefficient survive explicit
# population and GDP-per-capita controls? Reads WB covariates_wb.csv. -> 03_figures/cov_robust.csv
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
chl=chl.merge(lag(Wn,'Wo'),on=['iso3','year']).merge(lag(Wgeo,'Wg'),on=['iso3','year'])
cov=pd.read_csv(D+"covariates_wb.csv")
cov['lpop']=np.log(cov['pop']); cov['lgdp']=np.log(cov.gdp_pc)
chl=chl.merge(cov[['iso3','year','lpop','lgdp']],on=['iso3','year'],how='left')

def run(cols,label):
    d=chl.set_index(['iso3','year'])[['lchl','Wo','Wg']+[c for c in cols if c not in('Wo','Wg')]].dropna()
    d=d.rename(columns={'lchl':'y'})
    r=PanelOLS(d['y'],sm.add_constant(d[cols]),entity_effects=True,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
    ci=r.conf_int()
    print(f"{label:34s} ocean {r.params['Wo']:+.3f}{sig(r.pvalues['Wo'])} [{ci.loc['Wo','lower']:.3f},{ci.loc['Wo','upper']:.3f}] | "
          f"geo {r.params['Wg']:+.3f}{sig(r.pvalues['Wg'])} | n={len(d)}")
    return {'spec':label,'lambda_ocean':round(float(r.params['Wo']),4),'ci_lo':round(float(ci.loc['Wo','lower']),4),
            'ci_hi':round(float(ci.loc['Wo','upper']),4),'p':float(r.pvalues['Wo']),
            'lambda_geo':round(float(r.params['Wg']),4),'n':len(d)}
rows=[run(['Wo','Wg'],'Baseline (two-way FE)'),
      run(['Wo','Wg','lpop'],'+ log population'),
      run(['Wo','Wg','lgdp'],'+ log GDP per capita'),
      run(['Wo','Wg','lpop','lgdp'],'+ both covariates')]
pd.DataFrame(rows).to_csv('03_figures/cov_robust.csv',index=False)
print('\nsaved 03_figures/cov_robust.csv')
