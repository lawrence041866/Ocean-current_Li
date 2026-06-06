# Louvain community detection on the ocean-connectivity network + within- vs between-cluster pollution-synchrony test
# Output: 11 clusters Q=0.80; within 0.18 vs between 0.02 = 7.4x, p<1e-38 (manuscript Fig. 4)
import pandas as pd, numpy as np, networkx as nx, community as cl, warnings
from scipy import stats
warnings.filterwarnings('ignore')
D="01_data/"
W=pd.read_csv(D+"W_sovereign_directed.csv")
chl=pd.read_csv(D+"chl_yearly_2003_2021.csv"); chl['year']=chl.year.astype(int)
chl=chl.groupby(['iso3','year']).apply(lambda g:np.average(g.chlor_a_mean,weights=g.pixel_count)).reset_index(name='chl')
chl['lchl']=np.log(chl.chl)
countries=sorted(set(chl.iso3)&(set(W.src)|set(W.dst)))
G=nx.Graph()
for c in countries: G.add_node(c)
for r in W.itertuples():
    if r.src in countries and r.dst in countries and r.src!=r.dst:
        if G.has_edge(r.src,r.dst): G[r.src][r.dst]['weight']+=r.n_events
        else: G.add_edge(r.src,r.dst,weight=r.n_events)
part=cl.best_partition(G,weight='weight',random_state=42)
print(f"n_clusters={len(set(part.values()))} modularity Q={cl.modularity(part,G,weight='weight'):.3f}")
piv=chl[chl.iso3.isin(countries)].pivot(index='year',columns='iso3',values='lchl')
anom=piv.apply(lambda c: c-np.polyval(np.polyfit(piv.index[c.notna()],c[c.notna()],1),piv.index) if c.notna().sum()>3 else c)
pin,pout=[],[]
cl_list=list(countries)
for i in range(len(cl_list)):
    for j in range(i+1,len(cl_list)):
        a,b=cl_list[i],cl_list[j]
        if a in anom and b in anom:
            dd=pd.DataFrame({'a':anom[a],'b':anom[b]}).dropna()
            if len(dd)>=10:
                r=np.corrcoef(dd.a,dd.b)[0,1]
                (pin if part[a]==part[b] else pout).append(r)
p=stats.mannwhitneyu(pin,pout,alternative='greater')[1]
print(f"within={np.mean(pin):.3f} between={np.mean(pout):.3f} = {np.mean(pin)/np.mean(pout):.1f}x p={p:.1e}")
