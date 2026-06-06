# Robustness: re-run the main regression with alternative OHI outcomes (Clean Waters / Overall index / Biodiversity)
# Test whether ocean connectivity stays independently significant on an independent dataset (manuscript Fig. 2a + SI Table S3)
# Output: Clean Waters ocean 0.64*** (geography -0.16 ns); Overall index / Biodiversity are geography-dominated
import pandas as pd, numpy as np, warnings, country_converter as coco
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
sig=lambda p:'***' if p<.01 else '**' if p<.05 else '*' if p<.1 else ''
D="01_data/"

W=pd.read_csv(D+"W_sovereign_directed.csv")
hx=pd.read_csv(D+"hex2country.csv")
ohi=pd.read_csv(D+"OHI_panel.csv")
ohi['iso3']=coco.convert(ohi.region_name.tolist(),src='name',to='ISO3',not_found=None)
ohi=ohi[ohi.iso3.notna()&(ohi.iso3!='not found')]

def build_W(countries):
    idx={c:i for i,c in enumerate(countries)};N=len(countries)
    Wsym=np.zeros((N,N))
    for r in W.itertuples():
        if r.src in idx and r.dst in idx and r.src!=r.dst:
            Wsym[idx[r.src],idx[r.dst]]+=r.n_events; Wsym[idx[r.dst],idx[r.src]]+=r.n_events
    cen=hx.groupby('iso3')[['lon','lat']].median().reindex(countries)
    def hav(a,b):
        la1,lo1,la2,lo2=map(np.radians,[a[1],a[0],b[1],b[0]])
        return 2*6371*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
    Wgeo=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i!=j and cen.iloc[i].notna().all() and cen.iloc[j].notna().all():
                d=hav(cen.iloc[i].values,cen.iloc[j].values)
                if 0<d<3000: Wgeo[i,j]=1/d
    rn=lambda M:(M/np.where(M.sum(1,keepdims=True)==0,1,M.sum(1,keepdims=True)))
    return rn(Wsym),rn(Wgeo)

def run(panel,vcol,label):
    countries=sorted(set(panel.iso3)&(set(W.src)|set(W.dst)))
    Wn,Wg=build_W(countries)
    p=panel[panel.iso3.isin(countries)].copy()
    def lag(M,name):
        out=[]
        for yr,g in p.groupby('year'):
            g=g.set_index('iso3').reindex(countries)
            y=g[vcol].values.astype(float);ym=np.nan_to_num(y,nan=np.nanmean(y))
            out.append(pd.DataFrame({'iso3':countries,'year':yr,name:M@ym}))
        return pd.concat(out)
    p=p.merge(lag(Wn,'Wo'),on=['iso3','year']).merge(lag(Wg,'Wg'),on=['iso3','year'])
    d=p.set_index(['iso3','year'])[[vcol,'Wo','Wg']].dropna(); d.columns=['y','Wo','Wg']
    r=PanelOLS(d['y'],sm.add_constant(d[['Wo','Wg']]),entity_effects=True,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
    print(f"{label}: ocean beta={r.params['Wo']:+.3f}{sig(r.pvalues['Wo'])} | geo beta={r.params['Wg']:+.3f}{sig(r.pvalues['Wg'])} | n_countries={d.index.get_level_values(0).nunique()}")

ohi['lcw']=np.log(ohi.CleanWaters_14_1+1); run(ohi,'lcw','OHI Clean Waters')
ohi['lidx']=np.log(ohi.OHI_Index+1); run(ohi,'lidx','OHI Overall      ')
ohi['lbd']=np.log(ohi.Biodiversity+1); run(ohi,'lbd','OHI Biodiversity ')
