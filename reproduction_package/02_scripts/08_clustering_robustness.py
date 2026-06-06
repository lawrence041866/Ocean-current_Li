# Clustering robustness for the 11 circulation-defined pollution clusters
# Produces Supplementary Table (resolution sweep, seed stability, Leiden agreement,
# modularity significance vs degree-preserving null). Reads 01_data/W_sovereign_directed.csv.
import pandas as pd, numpy as np, networkx as nx, community as cl, warnings, random
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
warnings.filterwarnings('ignore')
rng = np.random.default_rng(0)
D = "01_data/"

# ---- build the same weighted connectivity graph as script 03 ----
W = pd.read_csv(D+"W_sovereign_directed.csv")
chl = pd.read_csv(D+"chl_yearly_2003_2021.csv")
countries = sorted(set(chl.iso3) & (set(W.src) | set(W.dst)))
G = nx.Graph()
G.add_nodes_from(countries)
for r in W.itertuples():
    if r.src in countries and r.dst in countries and r.src != r.dst:
        if G.has_edge(r.src, r.dst): G[r.src][r.dst]['weight'] += r.n_events
        else: G.add_edge(r.src, r.dst, weight=r.n_events)
print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
      f"density {nx.density(G):.3f}")

ref = cl.best_partition(G, weight='weight', random_state=42)
ref_k = len(set(ref.values()))
ref_Q = cl.modularity(ref, G, weight='weight')
nodes = list(countries)
def vec(part): return [part[n] for n in nodes]
print(f"\nReference (Louvain, seed 42): k={ref_k}, Q={ref_Q:.3f}")

# ---- 1. resolution sweep ----
print("\n[1] Resolution sweep (Louvain)")
print("gamma  k   Q     ARI_vs_ref")
for gamma in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]:
    p = cl.best_partition(G, weight='weight', resolution=gamma, random_state=42)
    k = len(set(p.values())); Q = cl.modularity(p, G, weight='weight')
    ari = adjusted_rand_score(vec(ref), vec(p))
    print(f"{gamma:4.2f}  {k:2d}  {Q:.3f}  {ari:.3f}")

# ---- 2. seed stability (100 seeds at default resolution) ----
print("\n[2] Seed stability (100 random seeds, gamma=1.0)")
ks, Qs, aris = [], [], []
for s in range(100):
    p = cl.best_partition(G, weight='weight', random_state=s)
    ks.append(len(set(p.values()))); Qs.append(cl.modularity(p, G, weight='weight'))
    aris.append(adjusted_rand_score(vec(ref), vec(p)))
ks = np.array(ks)
print(f"k: mean {ks.mean():.2f}, mode {int(pd.Series(ks).mode()[0])}, "
      f"range [{ks.min()},{ks.max()}], %=11: {(ks==11).mean()*100:.0f}%")
print(f"Q: {np.mean(Qs):.3f} ± {np.std(Qs):.3f}")
print(f"ARI vs ref: {np.mean(aris):.3f} ± {np.std(aris):.3f} (min {np.min(aris):.3f})")

# ---- 3. Leiden agreement ----
print("\n[3] Leiden (igraph) vs Louvain")
try:
    import igraph as ig, leidenalg
    idx = {n: i for i, n in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in G.edges()]
    wts = [G[u][v]['weight'] for u, v in G.edges()]
    g = ig.Graph(n=len(nodes), edges=edges); g.es['weight'] = wts
    leid = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                    weights='weight', seed=42)
    lp = {nodes[i]: leid.membership[i] for i in range(len(nodes))}
    k = len(set(lp.values())); Q = cl.modularity(lp, G, weight='weight')
    print(f"Leiden: k={k}, Q={Q:.3f}, ARI vs Louvain={adjusted_rand_score(vec(ref),vec(lp)):.3f}, "
          f"NMI={normalized_mutual_info_score(vec(ref),vec(lp)):.3f}")
except Exception as e:
    print("Leiden unavailable:", e)

# ---- 4. modularity significance vs degree-preserving null ----
print("\n[4] Modularity significance (degree-preserving double-edge-swap null, 200 reps)")
nullQ = []
for _ in range(200):
    H = G.copy()
    try:
        nx.double_edge_swap(H, nswap=H.number_of_edges()*5, max_tries=H.number_of_edges()*50)
    except nx.NetworkXAlgorithmError:
        pass
    # keep original weight multiset, reassign at random to swapped edges
    w = [d['weight'] for *_ , d in G.edges(data=True)]
    random.shuffle(w)
    for (u, v), ww in zip(H.edges(), w): H[u][v]['weight'] = ww
    pn = cl.best_partition(H, weight='weight', random_state=0)
    nullQ.append(cl.modularity(pn, H, weight='weight'))
nullQ = np.array(nullQ)
z = (ref_Q - nullQ.mean()) / nullQ.std()
emp_p = (1 + (nullQ >= ref_Q).sum()) / (len(nullQ) + 1)
print(f"observed Q={ref_Q:.3f}; null Q={nullQ.mean():.3f} ± {nullQ.std():.3f}; "
      f"z={z:.1f}; empirical p={emp_p:.3g}")
