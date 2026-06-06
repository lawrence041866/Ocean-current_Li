# Plot: confounding forest plot (manuscript Fig. 3)
# panel a spatial-channel horse-race (ocean vs geography, 4 specs); panel b thermal/upwelling/detrend robustness (ocean, 5 specs)
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
import pyhdfe
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D="01_data/"

# ---------- shared: chl, Wn, Wgeo ----------
def avg(g): return np.average(g.chlor_a_mean,weights=g.pixel_count)
chl0=pd.read_csv(D+"chl_yearly_2003_2021.csv"); chl0['year']=chl0.year.astype(int)
chl0=chl0.groupby(['iso3','year'],as_index=False).apply(lambda g:pd.Series({'chl':avg(g)}))
chl0['lchl']=np.log(chl0.chl)
W=pd.read_csv(D+"W_sovereign_directed.csv")
countries=sorted(set(chl0.iso3)&(set(W.src)|set(W.dst)))
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
def basin(c):
    lo,la=cen.loc[c,'lon'],cen.loc[c,'lat']
    if pd.isna(lo): return 'Other'
    if -7<=lo<=42 and 30<=la<=48: return 'Med-Black'
    if -30<=lo<=35 and la>48: return 'NE.Atl-Eur'
    if -85<=lo<=-40 and la>=30: return 'NW.Atl'
    if -98<=lo<=-58 and 7<=la<30: return 'Carib-Gulf'
    if -70<=lo<=-30 and -60<=la<7: return 'SW.Atl'
    if -30<=lo<20 and -40<=la<30: return 'E.Atl-WAfr'
    if 20<=lo<100 and -45<=la<30: return 'Indian'
    if 100<=lo<=150 and la>=25: return 'NW.Pac'
    if 95<=lo<=180 and -50<=la<25: return 'SE.Asia-WPac'
    if lo<=-65: return 'E.Pac-Amer'
    if lo>150 or lo<-150: return 'Pac-Is'
    return 'Other'
ov={'USA':'NW.Atl','CAN':'NW.Atl','RUS':'NW.Pac','MEX':'Carib-Gulf','AUS':'SE.Asia-WPac','ZAF':'Indian','FRA':'NE.Atl-Eur'}
bmap={c:ov.get(c,basin(c)) for c in countries}
Wb=np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if i!=j and bmap[countries[i]]==bmap[countries[j]]: Wb[i,j]=1
Wb=rn(Wb)
def lagdf(df,M,name,col):
    piv=df.pivot(index='iso3',columns='year',values=col).reindex(countries); out=[]
    for yr in sorted(df.year.unique()):
        y=piv[yr].values.astype(float); ym=np.nan_to_num(y,nan=np.nanmean(y))
        out.append(pd.DataFrame({'iso3':countries,'year':yr,name:M@ym}))
    return pd.concat(out)

# ---------- panel a: horse-race ----------
chl=chl0[chl0.iso3.isin(countries)].copy()
chl=chl.merge(lagdf(chl,Wn,'Wo','lchl'),on=['iso3','year']).merge(lagdf(chl,Wgeo,'Wg','lchl'),on=['iso3','year']).merge(lagdf(chl,Wb,'Wb','lchl'),on=['iso3','year'])
clu=pd.read_csv(D+"c1_clusters.csv"); chl=chl.merge(clu,on='iso3',how='left')
chl['by']=chl.iso3.map(bmap)+'_'+chl.year.astype(str); chl['cy']=chl.cluster.astype(str)+'_'+chl.year.astype(str)
d=chl.set_index(['iso3','year'])
def pe(r,n): c=r.conf_int().loc[n]; return r.params[n],c.iloc[0],c.iloc[1]
def fe(cols):
    return PanelOLS(d['lchl'],sm.add_constant(d[cols]),entity_effects=True,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
rA=fe(['Wo','Wg']); rB=fe(['Wo','Wg','Wb'])
def hd(fes):
    dd=chl.dropna(subset=['lchl','Wo','Wg']+fes).copy()
    ids=np.column_stack([dd[c].astype('category').cat.codes.values for c in fes])
    Y=pyhdfe.create(ids,drop_singletons=False).residualize(dd[['lchl','Wo','Wg']].values)
    m=sm.OLS(Y[:,0],Y[:,1:]).fit(cov_type='cluster',cov_kwds={'groups':dd.iso3.values})
    ci=m.conf_int(); return (m.params[0],ci[0,0],ci[0,1]),(m.params[1],ci[1,0],ci[1,1])
C_o,C_g=hd(['iso3','by']); D_o,D_g=hd(['iso3','cy'])
panelA_labels=['Baseline\n(country+year FE)','+ Same-basin weight','+ Basin×year FE','+ Cluster×year FE']
A_ocean=[pe(rA,'Wo'),pe(rB,'Wo'),C_o,D_o]
A_geo  =[pe(rA,'Wg'),pe(rB,'Wg'),C_g,D_g]

# ---------- panel b: thermal / upwelling / detrend ----------
mhw=pd.read_csv("01_data/MHW_days_by_EEZ_yearly.csv")
mhw['year']=mhw.year.astype(int); mhw=mhw.groupby(['iso3','year'],as_index=False)['mhw_days'].mean()
c2=chl0.merge(mhw,on=['iso3','year'],how='inner'); c2=c2[c2.iso3.isin(countries)].drop_duplicates(['iso3','year']).copy()
c2['mhw_z']=(c2.mhw_days-c2.mhw_days.mean())/c2.mhw_days.std()
c2=c2.merge(lagdf(c2,Wn,'Wo','lchl'),on=['iso3','year']).merge(lagdf(c2,Wgeo,'Wg','lchl'),on=['iso3','year']).merge(lagdf(c2,Wn,'W_mhw','mhw_z'),on=['iso3','year'])
d2=c2.set_index(['iso3','year'])
def fe2(df,cols,ent=True):
    return PanelOLS(df['lchl' if 'lchl' in df.columns else df.columns[0]],sm.add_constant(df[cols]),entity_effects=ent,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
m0=PanelOLS(d2['lchl'],sm.add_constant(d2[['Wo','Wg']]),entity_effects=True,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
m1=PanelOLS(d2['lchl'],sm.add_constant(d2[['Wo','Wg','mhw_z']]),entity_effects=True,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
m2=PanelOLS(d2['lchl'],sm.add_constant(d2[['Wo','Wg','mhw_z','W_mhw']]),entity_effects=True,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
ebus={'PER','CHL','ECU','NAM','AGO','ZAF','MRT','MAR','SEN','GMB','SOM','OMN','USA','MEX'}
de=c2[~c2.iso3.isin(ebus)].set_index(['iso3','year'])
m3=PanelOLS(de['lchl'],sm.add_constant(de[['Wo','Wg']]),entity_effects=True,time_effects=True).fit(cov_type='clustered',cluster_entity=True)
cc=c2.sort_values(['iso3','year']).copy()
for c in ['lchl','Wo','Wg']: cc['d_'+c]=cc.groupby('iso3')[c].diff()
cc=cc.dropna(subset=['d_lchl','d_Wo','d_Wg']); dd4=cc.set_index(['iso3','year'])
m4=PanelOLS(dd4['d_lchl'],sm.add_constant(dd4[['d_Wo','d_Wg']]),time_effects=True).fit(cov_type='clustered',cluster_entity=True)
panelB_labels=['Baseline (2003–19)','+ Own marine heatwave','+ Partner heatwave','− Upwelling nations','First-difference']
B_ocean=[pe(m0,'Wo'),pe(m1,'Wo'),pe(m2,'Wo'),pe(m3,'Wo'),pe(m4,'d_Wo')]

for lab,v in zip(panelA_labels,A_ocean): print('A',lab.replace(chr(10),' '),round(v[0],3),[round(v[1],3),round(v[2],3)])
for lab,v in zip(panelB_labels,B_ocean): print('B',lab,round(v[0],3),[round(v[1],3),round(v[2],3)])
# dump Fig 2 source values
import pandas as _pd
_rows=[]
for lab,v in zip(panelA_labels,A_ocean): _rows.append({'panel':'2a','spec':lab.replace(chr(10),' '),'channel':'ocean','coef':round(v[0],4),'ci_lo':round(v[1],4),'ci_hi':round(v[2],4)})
for lab,v in zip(panelA_labels,A_geo):   _rows.append({'panel':'2a','spec':lab.replace(chr(10),' '),'channel':'geo','coef':round(v[0],4),'ci_lo':round(v[1],4),'ci_hi':round(v[2],4)})
for lab,v in zip(panelB_labels,B_ocean): _rows.append({'panel':'2b','spec':lab,'channel':'ocean','coef':round(v[0],4),'ci_lo':round(v[1],4),'ci_hi':round(v[2],4)})
_pd.DataFrame(_rows).to_csv('03_figures/fig2_coef_ci.csv',index=False)
print('fig2_coef_ci.csv written')

# ---------- plotting ----------
OC='#1f5f8b'; GC='#bdbdbd'
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(9.6,4.3))
def forest(ax,labels,ocean,geo=None,title=''):
    y=np.arange(len(labels))[::-1]
    for k,(c,lo,hi) in enumerate(ocean):
        yy=y[k]+(0.16 if geo else 0)
        ax.errorbar(c,yy,xerr=[[c-lo],[hi-c]],fmt='o',color=OC,ms=6,capsize=3,lw=1.6,zorder=3)
    if geo:
        for k,(c,lo,hi) in enumerate(geo):
            ax.errorbar(c,y[k]-0.16,xerr=[[c-lo],[hi-c]],fmt='s',color=GC,ms=5,capsize=3,lw=1.4,zorder=2)
    ax.axvline(0,ls='--',color='#999',lw=1)
    ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=8.5)
    ax.set_xlabel('Spatial spillover coefficient',fontsize=9)
    ax.set_title(title,fontsize=10,loc='left',fontweight='bold')
    ax.tick_params(labelsize=8.5); ax.set_axisbelow(True); ax.grid(axis='x',color='#eee')
    for s in ['top','right']: ax.spines[s].set_visible(False)
forest(ax1,panelA_labels,A_ocean,A_geo,'a   Spatial-channel horse-race')
forest(ax2,panelB_labels,B_ocean,None,'b   Thermal & upwelling controls')
from matplotlib.lines import Line2D
ax1.legend(handles=[Line2D([],[],marker='o',color=OC,ls='',ms=6,label='Ocean-current'),
                    Line2D([],[],marker='s',color=GC,ls='',ms=5,label='Geographic')],
           fontsize=8,loc='lower right',frameon=False)
ax2.set_xlim(-0.05,0.6)
fig.tight_layout()
fig.savefig('03_figures/Fig_ruling_out_confounding.png',dpi=300,bbox_inches='tight')
fig.savefig('03_figures/Fig_ruling_out_confounding.pdf',bbox_inches='tight')
print('FIG SAVED (PNG+PDF)')
