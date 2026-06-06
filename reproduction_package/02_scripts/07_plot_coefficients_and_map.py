# Re-draw Fig 1 (coefficient plot, with 95% CIs) and Fig 3 (global cluster map,
# colour-blind-safe palette + numbered on-map labels + within-cluster connectivity edges).
# Exports both PNG (300 dpi) and vector PDF. Reads fig1_coef_ci.csv (script 09) + cluster data.
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
D = "01_data/"; OUT = "03_figures/"

def save(fig, stem):
    fig.savefig(OUT+stem+".png", dpi=300, bbox_inches='tight')
    fig.savefig(OUT+stem+".pdf", bbox_inches='tight')
    plt.close(fig)

# ================== Fig 1 ==================
ci = pd.read_csv(OUT+"fig1_coef_ci.csv")
def get(outcome, ch):
    r = ci[(ci.outcome == outcome) & (ci.channel == ch)].iloc[0]
    return r.coef, r.ci_lo, r.ci_hi, r.p
outs = ['Chlorophyll-a (SDG 14.1.1)', 'OHI Clean Waters', 'OHI Index (overall)', 'OHI Biodiversity']
labels = ['Chlorophyll-a\n(SDG 14.1.1)', 'OHI Clean\nWaters', 'OHI Index\n(overall)', 'OHI\nBiodiversity']
def star(p): return '***' if p < .01 else '**' if p < .05 else '*' if p < .1 else 'n.s.'
oc = [get(o, 'ocean') for o in outs]; ge = [get(o, 'geo') for o in outs]

OC = '#0072B2'; GC = '#b9b9b9'; GR = '#009E73'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 4.6), gridspec_kw={'width_ratios': [1.55, 1]})
x = np.arange(len(labels)); w = 0.38
def err(vals): return [[v[0]-v[1] for v in vals], [v[2]-v[0] for v in vals]]
ax1.bar(x-w/2, [v[0] for v in oc], w, color=OC, label='Ocean-current connectivity', zorder=3,
        yerr=err(oc), capsize=3, error_kw={'lw': 1.1, 'ecolor': '#333'})
ax1.bar(x+w/2, [v[0] for v in ge], w, color=GC, label='Geographic proximity', zorder=3,
        yerr=err(ge), capsize=3, error_kw={'lw': 1.1, 'ecolor': '#777'})
for xi, v in zip(x-w/2, oc):
    ax1.text(xi, v[2]+0.03, star(v[3]), ha='center', va='bottom', fontsize=9, color=OC, fontweight='bold')
for xi, v in zip(x+w/2, ge):
    ax1.text(xi, v[2]+0.03, star(v[3]), ha='center', va='bottom', fontsize=9, color='#666')
ax1.axhline(0, color='#333', lw=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel('Spatial spillover coefficient (lambda)', fontsize=10)
ax1.text(-0.02, 1.04, 'a', transform=ax1.transAxes, fontsize=14, fontweight='bold', va='top')
ax1.legend(fontsize=8.5, loc='upper right', frameon=False)
ax1.set_ylim(-0.62, 1.18); ax1.grid(axis='y', color='#eee', zorder=0)
for sp in ['top', 'right']: ax1.spines[sp].set_visible(False)

ols, ols_lo, ols_hi, _ = get('OLS (ocean-only, chl)', 'ocean')
tsls, tsls_lo, tsls_hi = 0.98, 0.63, 1.32   # spatial-2SLS (script 02 / Table S4)
xs = [0, 1]; vals = [ols, tsls]
ax2.bar(xs, vals, 0.55, color=[OC, GR], zorder=3,
        yerr=[[ols-ols_lo, tsls-tsls_lo], [ols_hi-ols, tsls_hi-tsls]],
        capsize=5, error_kw={'lw': 1.4, 'ecolor': '#333'})
for xi, v, hi in zip(xs, vals, [ols_hi, tsls_hi]):
    ax2.text(xi, hi+0.04, f'{v:.2f}***', ha='center', fontsize=10, fontweight='bold')
ax2.set_xticks(xs); ax2.set_xticklabels(['OLS', 'Spatial 2SLS\n(endogeneity-corrected)'], fontsize=9)
ax2.set_ylabel('Connectivity coefficient (lambda)', fontsize=10)
ax2.text(-0.04, 1.04, 'b', transform=ax2.transAxes, fontsize=14, fontweight='bold', va='top')
ax2.set_ylim(0, 1.55); ax2.grid(axis='y', color='#eee', zorder=0)
for sp in ['top', 'right']: ax2.spines[sp].set_visible(False)
fig.tight_layout()
save(fig, "C1_core_results")
print("Fig1 saved (with 95% CI, PNG+PDF)")

# ================== Fig 3 (dot style) ==================
hx = pd.read_csv(D+"hex2country.csv")
clu = pd.read_csv(D+"c1_clusters.csv")
hx = hx.merge(clu, on='iso3', how='left')
W = pd.read_csv(D+"W_sovereign_directed.csv")
cen = hx.dropna(subset=['cluster']).groupby('iso3')[['lon', 'lat']].median()
iso_clu = clu.set_index('iso3').cluster.to_dict()

names = {0: 'W Indian Ocean / Arabian Sea', 1: 'Mediterranean', 2: 'NW Africa',
         3: 'SW/SE Atlantic & SW Pacific', 4: 'Wider Caribbean', 5: 'SE Asia / W Pacific',
         6: 'NE Atlantic / North & Baltic', 7: 'W Africa / Gulf of Guinea',
         8: 'N Indian Ocean / Bay of Bengal', 9: 'NW Pacific (+ Black Sea)',
         10: 'Gulf of Mexico / Central America'}
palette = {0: '#D55E00', 1: '#0072B2', 2: '#882255', 3: '#56B4E9', 4: '#CC79A7',
           5: '#009E73', 6: '#44AA99', 7: '#E69F00', 8: '#999933', 9: '#332288',
           10: '#F0E442'}
order = [1, 6, 0, 8, 5, 4, 10, 9, 7, 3, 2]          # legend order (large -> small)

fig, ax = plt.subplots(figsize=(12, 6.4))
# faint world coastline basemap (all coastal cells, light grey)
ax.scatter(hx.lon, hx.lat, s=0.9, c='#aeb7c1', linewidths=0, zorder=1)
# within-cluster connectivity edges (skip antimeridian-wrapping pairs)
for r in W.itertuples():
    a, b = r.src, r.dst
    if a in iso_clu and b in iso_clu and a in cen.index and b in cen.index and iso_clu[a] == iso_clu[b]:
        lo1, la1 = cen.loc[a, 'lon'], cen.loc[a, 'lat']
        lo2, la2 = cen.loc[b, 'lon'], cen.loc[b, 'lat']
        if abs(lo1 - lo2) > 180:
            continue
        ax.plot([lo1, lo2], [la1, la2], color=palette[iso_clu[a]], lw=0.5, alpha=0.30, zorder=2)
# one dot per nation at its coastal centroid
for c in order:
    iso = [i for i in cen.index if iso_clu[i] == c]
    ax.scatter(cen.loc[iso, 'lon'], cen.loc[iso, 'lat'], s=58, c=palette[c],
               edgecolors='white', linewidths=0.6, zorder=4)
handles = [Line2D([], [], marker='o', ls='', ms=8, color=palette[c], markeredgecolor='white',
                  label=f'{names[c]} (n={(clu.cluster==c).sum()})') for c in order]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.075), fontsize=8.5,
          ncol=4, frameon=False, title='Eleven current-defined marine-pollution clusters',
          title_fontsize=10, handletextpad=0.4, columnspacing=1.4)
ax.set_xlim(-180, 185); ax.set_ylim(-72, 90)
ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect(1.25)
for sp in ax.spines.values(): sp.set_visible(False)
save(fig, "C1_cluster_map")
print("Fig3 saved (dot style, colour-blind palette, PNG+PDF)")
print("cluster counts:", dict(clu.cluster.value_counts().sort_index()))
