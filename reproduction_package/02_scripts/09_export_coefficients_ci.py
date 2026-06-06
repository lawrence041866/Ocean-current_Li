# Export point estimates + 95% CIs for Fig 1 (and Source Data).
# Reproduces the spatial-lag models of scripts 01 / 01b / 02 and writes 03_figures/fig1_coef_ci.csv
import pandas as pd, numpy as np, warnings, country_converter as coco
warnings.filterwarnings('ignore')
from linearmodels.panel import PanelOLS
import statsmodels.api as sm
D = "01_data/"
W = pd.read_csv(D+"W_sovereign_directed.csv")
hx = pd.read_csv(D+"hex2country.csv")

def build_W(countries):
    idx = {c: i for i, c in enumerate(countries)}; N = len(countries)
    Wsym = np.zeros((N, N))
    for r in W.itertuples():
        if r.src in idx and r.dst in idx and r.src != r.dst:
            Wsym[idx[r.src], idx[r.dst]] += r.n_events; Wsym[idx[r.dst], idx[r.src]] += r.n_events
    cen = hx.groupby('iso3')[['lon', 'lat']].median().reindex(countries)
    def hav(a, b):
        la1, lo1, la2, lo2 = map(np.radians, [a[1], a[0], b[1], b[0]])
        return 2*6371*np.arcsin(np.sqrt(np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2))
    Wgeo = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and cen.iloc[i].notna().all() and cen.iloc[j].notna().all():
                dd = hav(cen.iloc[i].values, cen.iloc[j].values)
                if 0 < dd < 3000: Wgeo[i, j] = 1/dd
    rn = lambda M: (M/np.where(M.sum(1, keepdims=True) == 0, 1, M.sum(1, keepdims=True)))
    return rn(Wsym), rn(Wgeo)

def fit(panel, vcol, two_channel=True):
    countries = sorted(set(panel.iso3) & (set(W.src) | set(W.dst)))
    Wn, Wg = build_W(countries)
    p = panel[panel.iso3.isin(countries)].copy()
    def lag(M, name):
        out = []
        for yr, g in p.groupby('year'):
            g = g.set_index('iso3').reindex(countries)
            y = g[vcol].values.astype(float); ym = np.nan_to_num(y, nan=np.nanmean(y))
            out.append(pd.DataFrame({'iso3': countries, 'year': yr, name: M@ym}))
        return pd.concat(out)
    p = p.merge(lag(Wn, 'Wo'), on=['iso3', 'year']).merge(lag(Wg, 'Wg'), on=['iso3', 'year'])
    cols = ['Wo', 'Wg'] if two_channel else ['Wo']
    d = p.set_index(['iso3', 'year'])[[vcol]+cols].dropna()
    d.columns = ['y']+cols
    r = PanelOLS(d['y'], sm.add_constant(d[cols]), entity_effects=True, time_effects=True).fit(
        cov_type='clustered', cluster_entity=True)
    ci = r.conf_int()
    return r, ci, len(d), d.index.get_level_values(0).nunique()

rows = []
def add(outcome, channel, r, ci, key, n, ncty):
    rows.append({'outcome': outcome, 'channel': channel,
                 'coef': round(float(r.params[key]), 4),
                 'ci_lo': round(float(ci.loc[key, 'lower']), 4),
                 'ci_hi': round(float(ci.loc[key, 'upper']), 4),
                 'p': float(r.pvalues[key]), 'n': n, 'n_countries': ncty})

# chlorophyll
chl = pd.read_csv(D+"chl_yearly_2003_2021.csv"); chl['year'] = chl.year.astype(int)
chl = chl.groupby(['iso3', 'year']).apply(lambda g: np.average(g.chlor_a_mean, weights=g.pixel_count)).reset_index(name='chl')
chl['lchl'] = np.log(chl.chl)
r, ci, n, nc = fit(chl, 'lchl')
add('Chlorophyll-a (SDG 14.1.1)', 'ocean', r, ci, 'Wo', n, nc)
add('Chlorophyll-a (SDG 14.1.1)', 'geo',   r, ci, 'Wg', n, nc)
# OLS ocean-only (Fig 1b left bar)
r1, ci1, n1, nc1 = fit(chl, 'lchl', two_channel=False)
add('OLS (ocean-only, chl)', 'ocean', r1, ci1, 'Wo', n1, nc1)

# OHI outcomes
ohi = pd.read_csv(D+"OHI_panel.csv")
ohi['iso3'] = coco.convert(ohi.region_name.tolist(), src='name', to='ISO3', not_found=None)
ohi = ohi[ohi.iso3.notna() & (ohi.iso3 != 'not found')]
for col, lab in [('CleanWaters_14_1', 'OHI Clean Waters'),
                 ('OHI_Index', 'OHI Index (overall)'),
                 ('Biodiversity', 'OHI Biodiversity')]:
    ohi['_l'] = np.log(ohi[col]+1)
    r, ci, n, nc = fit(ohi, '_l')
    add(lab, 'ocean', r, ci, 'Wo', n, nc)
    add(lab, 'geo',   r, ci, 'Wg', n, nc)

out = pd.DataFrame(rows)
out.to_csv("03_figures/fig1_coef_ci.csv", index=False)
pd.set_option('display.width', 140)
print(out.to_string(index=False))
print("\nsaved 03_figures/fig1_coef_ci.csv")
