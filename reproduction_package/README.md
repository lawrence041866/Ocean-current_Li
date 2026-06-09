# Reproduction package

**Ocean-current connectivity drives the transboundary synchrony of marine pollution across coastal nations**

Yiran Li (Graduate School of Economics, Kobe University)

This package reproduces every number, figure and supplementary table in the manuscript from public data.

---

## 1. Overview

We treat **ocean-current connectivity** as a spatial weight matrix and quantify the transboundary synchrony of marine pollution (SDG 14.1) across **123 coastal nations, 2003–2021**. The pipeline goes from public satellite/connectivity data → a national connectivity matrix *W* → panel spatial-econometric models → confounding tests → community detection → figures and tables.

## 2. Directory structure

```
.
├── 01_data/                          # input data (public; provenance below)
│   ├── chl_yearly_2003_2021.csv      # MODIS-Aqua chlorophyll-a by EEZ-year (main DV)
│   ├── OHI_panel.csv                 # Ocean Health Index goal scores (independent DV)
│   ├── W_sovereign_directed.csv      # coastalNet ocean-current connectivity (core weight)
│   ├── hex2country.csv               # 26,642 coastal cells → nation (lon/lat)
│   ├── c1_clusters.csv               # 11 Louvain clusters (nation → cluster)
│   ├── SI_clusters_members.csv       # cluster membership (Supplementary Table 5)
│   ├── covariates_wb.csv             # World Bank population & GDP per capita
│   └── MHW_days_by_EEZ_yearly.csv    # marine-heatwave days per EEZ-year (thermal control)
├── 02_scripts/                       # analysis scripts (Python / GEE JS)
├── 03_figures/                       # figures (PNG + vector PDF) + per-figure value CSVs
├── 04_documents/                     # manuscript and SI (.docx)
├── Source_Data.xlsx                  # numeric values behind every figure
├── LICENSE · CITATION.cff · .zenodo.json
└── README.md
```

## 3. Data sources and provenance

| Dataset | Source | Used for |
|---|---|---|
| MODIS-Aqua chlorophyll-a (L3) | NASA Ocean Biology Processing Group (via Google Earth Engine) | Main pollution proxy (SDG 14.1.1) |
| Ocean Health Index goal scores | ohi-science.org | Independent water-quality measure |
| coastalNet connectivity events | Assis et al., *Scientific Data* 12, 737 (2025) | Ocean-current weight matrix *W* |
| Maritime boundaries (EEZ v12) | Flanders Marine Institute, marineregions.org | EEZ aggregation |
| Marine-heatwave days per EEZ | Derived from NOAA OISST (Hobday et al. 2016 definition) | Thermal-confounding control |
| Population, GDP per capita | World Bank World Development Indicators | Covariate robustness |

All inputs are public; large raw inputs (8.7 GB coastalNet trajectories) are **not** included because `W_sovereign_directed.csv` is the aggregated product needed to reproduce results.

## 4. Scripts → outputs

| Script | Produces | Maps to |
|---|---|---|
| `00_gee_chlorophyll_export.js` | chlorophyll-a by EEZ-year (run in Google Earth Engine) | `chl_yearly_2003_2021.csv` |
| `01_main_regression_ocean_vs_geo.py` | ocean 0.44\*\*\* vs geography 0.38\*\*\* | **Fig. 2a** |
| `01b_robustness_ohi_outcomes.py` | Clean Waters 0.64\*\*\*; overall 0.05; biodiversity 0.24\* | **Fig. 2a / Table S3** |
| `02_endogeneity_spatial_2sls.py` | OLS 0.63\*\*\*; 2SLS 0.98\*\*\*, F = 96 | **Fig. 2b / Table S4** |
| `03_community_detection_clusters.py` | 11 clusters, Q = 0.80; within/between 7.4× | **Table S5 / `c1_clusters.csv`** |
| `04_mechanism_horserace_region_fe.py` | horse-race; basin×year & cluster×year FE | **Fig. 3a / Table S1** |
| `05_proxy_validity_heatwave_upwelling.py` | marine-heatwave controls; upwelling; first-diff | **Fig. 3b / Table S2** |
| `06_plot_confounding_forest.py` | confounding forest plot | **Fig. 3** |
| `07_plot_coefficients_and_map.py` | coefficient plot (Fig 2) & cluster map (Fig 4) | **Fig. 2 / Fig. 4** |
| `08_clustering_robustness.py` | resolution sweep, 100-seed stability, Leiden, modularity significance | **Table S6 / Note 3** |
| `09_export_coefficients_ci.py` | coefficients + 95% CIs for Fig 2 | `03_figures/fig1_coef_ci.csv` |
| `10_build_source_data.py` | per-figure numeric values | **`Source_Data.xlsx`** |
| `11_plot_concept_figure.py` | conceptual / workflow figure | **Fig. 1** |
| `12_covariate_robustness.py` | population & GDP-per-capita controls | **Table S7 / Note 4** |

## 5. How to run

```bash
pip install pandas numpy linearmodels statsmodels pyhdfe python-louvain networkx scipy country_converter scikit-learn leidenalg igraph openpyxl matplotlib
# run from the repository root
python3 02_scripts/01_main_regression_ocean_vs_geo.py
python3 02_scripts/01b_robustness_ohi_outcomes.py
python3 02_scripts/02_endogeneity_spatial_2sls.py
python3 02_scripts/03_community_detection_clusters.py
python3 02_scripts/04_mechanism_horserace_region_fe.py
python3 02_scripts/05_proxy_validity_heatwave_upwelling.py
python3 02_scripts/06_plot_confounding_forest.py
python3 02_scripts/09_export_coefficients_ci.py
python3 02_scripts/07_plot_coefficients_and_map.py
python3 02_scripts/08_clustering_robustness.py
python3 02_scripts/11_plot_concept_figure.py
python3 02_scripts/12_covariate_robustness.py
python3 02_scripts/10_build_source_data.py
```

All scripts read from `01_data/` and must be run from the repository root. (Run `09_export_coefficients_ci.py` before `07_plot_coefficients_and_map.py`, which uses its output.)

## 6. Key reproduced numbers

| Result | Value |
|---|---|
| Ocean vs geography (chl, two-way FE) | λ_ocean = 0.44\*\*\*, λ_geo = 0.38\*\*\* |
| Horse-race (basin×year FE) | λ_ocean = 0.44\*\*\*, λ_geo = 0.08 n.s. |
| Thermal control (partner heatwave) | λ_ocean = 0.46\*\*\* |
| First-difference | λ_ocean = 0.42\*\*\* |
| Spatial 2SLS | λ = 0.98\*\*\*, F = 96.2, 95% CI [0.63, 1.32] |
| Community detection | 11 clusters, Q = 0.80, within/between 7.4× (p < 10⁻³⁸) |
| Clustering robustness | 94% of seeds give 11 clusters; Leiden ≡ Louvain; modularity z = 6.6 |
| Covariate robustness | λ_ocean = 0.44\*\*\* with population + GDP controls |
| Cluster–basin concordance | 83.7% |

## 7. Citation, license, contact

If you use this package, please cite the paper (in review, 2026) and the coastalNet dataset (Assis et al., *Scientific Data*, 2025). Code released under the MIT License; data retain their original licenses. Contact: Yiran Li ([email to add]).

This package is archived on **Zenodo** (DOI assigned on release) and mirrored on GitHub.
