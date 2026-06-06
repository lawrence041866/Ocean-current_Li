# Conceptual / workflow figure (proposed new Fig 1) — data flow of the study.
# Exports C1_Fig1_concept.png (300 dpi) + .pdf (vector).
import numpy as np, warnings, matplotlib
warnings.filterwarnings('ignore'); matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

NAVY='#1F4E79'; BLUE='#0072B2'; TEAL='#117733'; GREY='#6b6b6b'; LGREY='#eef2f6'
plt.rcParams['font.family']='DejaVu Sans'
fig,ax=plt.subplots(figsize=(11.8,5.6)); ax.set_xlim(0,100); ax.set_ylim(0,60); ax.axis('off')

def box(x,y,w,h,fc,ec,lw=1.3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=1.6',
                fc=fc,ec=ec,lw=lw,zorder=2,mutation_aspect=0.55))
def txt(x,y,s,fs=9,c='#1a1a1a',w='normal',ha='center',it=False):
    ax.text(x,y,s,fontsize=fs,color=c,fontweight=w,ha=ha,va='center',style='italic' if it else 'normal',zorder=4)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=15,
                color=GREY,lw=1.6,zorder=1,shrinkA=0,shrinkB=0))

# stage headers
for sx,lab in [(13,'1  Physical inputs'),(37,'2  Connectivity weight W'),
               (62.5,'3  Spatial-econometric model'),(89,'4  Findings')]:
    txt(sx,57,lab,fs=10.5,c=NAVY,w='bold')

# ---- Stage 1: inputs (x 2..24) ----
box(2,40,22,11,LGREY,BLUE); txt(13,48.2,'Ocean-current connectivity',fs=9,w='bold',c=NAVY)
txt(13,44.0,'181.8M Lagrangian events\n26,642 coastal cells (coastalNet)',fs=8,c='#333')
box(2,25.5,22,10,LGREY,BLUE); txt(13,32.6,'Marine pollution',fs=9,w='bold',c=NAVY)
txt(13,28.8,'MODIS chlorophyll-a (SDG 14.1.1)\n+ OHI Clean Waters',fs=8,c='#333')
box(2,11,22,10,LGREY,BLUE); txt(13,18.1,'Confounding controls',fs=9,w='bold',c=NAVY)
txt(13,14.4,'Heatwaves, basins, upwelling,\npopulation, GDP, geography',fs=8,c='#333')

# ---- Stage 2: W heatmap (x 31.5..43.4) ----
gx,gy,gs=31.5,30.5,1.7; nn=7
rng=np.random.default_rng(3); M=rng.random((nn,nn)); M=(M+M.T)/2; np.fill_diagonal(M,1)
for i in range(nn):
    for j in range(nn):
        ax.add_patch(plt.Rectangle((gx+j*gs,gy+(nn-1-i)*gs),gs,gs,
                     fc=plt.cm.Blues(0.15+0.7*M[i,j]),ec='white',lw=0.4,zorder=2))
mxc=gx+nn*gs/2
txt(mxc,gy+nn*gs+2.4,'W : 123 × 123 nations',fs=8.5,w='bold',c=NAVY)
txt(mxc,gy-4.0,'symmetrised, row-standardised',fs=7.6,c=GREY,it=True)
txt(mxc,gy-7.6,'vs geographic &\nsame-basin weights',fs=7.8,c='#333')

# ---- Stage 3: model box (x 50..76, y 26..44) ----
box(50,26,26,18,'#f4f8fb',TEAL,lw=1.5)
txt(63,40.2,'Panel spatial-lag',fs=10,w='bold',c=TEAL)
txt(63,34.6,r'ln$\,$POLL$_{it}$ = c$_i$ + δ$_t$',fs=9.5)
txt(63,30.0,r'+ λ Σ$_j$ W$_{ij}$ ln$\,$POLL$_{jt}$ + Xβ',fs=9.5)
txt(63,21.5,'two-way FE · country-clustered SE\nbasin×year / cluster×year FE · 2SLS',fs=7.8,c=GREY)

# ---- Stage 4: findings (x 79..100) ----
def find(y,t,s):
    box(79,y,21,9,LGREY,BLUE,lw=1.1); txt(89.5,y+6.2,t,fs=8.6,w='bold',c=NAVY); txt(89.5,y+2.7,s,fs=7.5,c='#333')
find(40.5,'Currents > geography','λ = 0.44 vs 0.38; robust\nacross two pollution datasets')
find(27.5,'Specific to currents','survives region×year FE &\nheatwaves; silent placebo')
find(14.5,'11 governance clusters','Q = 0.80; within 7.4× between;\n83.7% basin-concordant')

# arrows: inputs -> W (start right of boxes at x=24.6, end left of matrix at x=30.8)
arrow(24.6,45.5,31,39); arrow(24.6,30.5,31,36); arrow(24.6,16.5,31,34.5)
# W -> model
arrow(43.8,36,49.8,35)
# model -> findings (start right of box x=76.2, end left of finding boxes x=78.8)
arrow(76.2,38,78.8,45); arrow(76.2,35,78.8,32); arrow(76.2,32,78.8,19)

fig.savefig('03_figures/C1_Fig1_concept.png',dpi=300,bbox_inches='tight')
fig.savefig('03_figures/C1_Fig1_concept.pdf',bbox_inches='tight')
print('concept figure regenerated (PNG+PDF)')
