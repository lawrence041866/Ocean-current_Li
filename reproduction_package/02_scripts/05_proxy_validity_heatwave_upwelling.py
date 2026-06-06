# Proxy validity: addressing the 'chlorophyll-a != pollution; driven by temperature/upwelling/seasonal natural variability' concern
# Control for marine heatwaves (MHW_days, climatology-removed SST stress anomalies) + drop upwelling-system nations + first-difference detrending
# Conclusion: the ocean coefficient is stable at 0.42-0.46*** across specifications; unchanged after controlling for connected-partner heatwaves
#       => pollution synchrony is not an artefact of shared warming/upwelling/trends
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
sig=lambda p:'***' if p<.01 else '**' if p<.05 else '*' if p<.1 else ' n.s.'
D="01_data/"
chl=pd.read_csv(D+"chl_yearly_2003_2021.csv"); chl['year']=chl.year.astype(int)
chl=chl.groupby(['iso3','year'],as_index=False).apply(lambda g:pd.Series({'chl':np.average(g.chlor_a_mean,weights=g.pixel_count)}))
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
            dd=hav(cen.iloc[i].values,cen.iloc[j].values)
            if 0<dd<3000: Wgeo[i,j]=1/dd
Wgeo=rn(Wgeo)
mhw=pd.read_csv("01_data/MHW_days_by_EEZ_yearly.csv")
mhw['year']=mhw.year.astype(int); mhw=mhw.groupby(['iso3','year'],as_index=False)['mhw_days'].mean()
chl=chl.merge(mhw,on=['iso3','year'],how='inner')                 # overlap period 2003-2019
chl=chl[chl.iso3.isin(countries)].drop_duplicates(['iso3','year']).copy()
chl['mhw_z']=(chl.mhw_days-chl.mhw_days.mean())/chl.mhw_days.std()
def lag(M,name,col):
    piv=chl.pivot(index='iso3',columns='year',values=col).reindex(countries); out=[]
    for yr in sorted(chl.year.unique()):
        y=piv[yr].values.astype(float); ym=np.nan_to_num(y,nan=np.nanmean(y))
        out.append(pd.DataFrame({'iso3':countries,'year':yr,name:M@ym}))
    return pd.concat(out)
chl=chl.merge(lag(Wn,'Wo','lchl'),on=['iso3','year']).merge(lag(Wgeo,'Wg','lchl'),on=['iso3','year']).merge(lag(Wn,'W_mhw','mhw_z'),on=['iso3','year'])
d=chl.set_index(['iso3','year'])
def fe(cols,label):
    r=PanelOLS(d['lchl'],sm.add_constant(d[cols]),entity_effects=True,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
    extra=" | ".join(f"{c} {r.params[c]:+.3f}{sig(r.pvalues[c])}" for c in cols if c not in['Wo','Wg'])
    print(f"{label}: ocean {r.params['Wo']:+.3f}{sig(r.pvalues['Wo'])} | geo {r.params['Wg']:+.3f}{sig(r.pvalues['Wg'])}"+(" | "+extra if extra else ""))
fe(['Wo','Wg'],"M0 baseline (2003-19 overlap) ")
fe(['Wo','Wg','mhw_z'],"M1 + own marine heatwave      ")
fe(['Wo','Wg','mhw_z','W_mhw'],"M2 + partner heatwave (core)  ")
ebus={'PER','CHL','ECU','NAM','AGO','ZAF','MRT','MAR','SEN','GMB','SOM','OMN','USA','MEX'}
de=chl[~chl.iso3.isin(ebus)].set_index(['iso3','year'])
re=PanelOLS(de['lchl'],sm.add_constant(de[['Wo','Wg']]),entity_effects=True,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
print(f"M3 drop upwelling nations (n={de.index.get_level_values(0).nunique()})   : ocean {re.params['Wo']:+.3f}{sig(re.pvalues['Wo'])} | geo {re.params['Wg']:+.3f}{sig(re.pvalues['Wg'])}")
ch=chl.sort_values(['iso3','year']).copy()
for c in ['lchl','Wo','Wg']: ch['d_'+c]=ch.groupby('iso3')[c].diff()
ch=ch.dropna(subset=['d_lchl','d_Wo','d_Wg']); dd=ch.set_index(['iso3','year'])
rd=PanelOLS(dd['d_lchl'],sm.add_constant(dd[['d_Wo','d_Wg']]),time_effects=True).fit(cov_type='clustered',cluster_entity=True)
print(f"M4 first-difference (detrend) : ocean {rd.params['d_Wo']:+.3f}{sig(rd.pvalues['d_Wo'])} | geo {rd.params['d_Wg']:+.3f}{sig(rd.pvalues['d_Wg'])}")
