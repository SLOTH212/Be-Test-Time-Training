import os,sys,json,csv,hashlib
os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg',MPLCONFIGDIR='/tmp/r2_cpu_mplconfig')
sys.path.insert(0,'/tmp/r2_cpu_plot_deps')
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
P=Path(__file__).resolve().parent;R=Path('/home/zonghan/ttt/runs/formal/mechanism_old1k_v2_20260921');A=['OFF','L0','L6','L12','L18','L24','ALL'];cp={x['sample_id']:x for x in json.loads((R/'COHORT.json').read_text())};prov=json.loads((P/'input_provenance.json').read_text());plt.rcParams.update({'font.size':8,'pdf.fonttype':42})
start=int(sys.argv[1]);stop=min(start+20,170)
with PdfPages(P/'figures'/f'curves_part_{start:03d}.pdf') as pdf:
 for i,sid in enumerate(sorted(cp)[start:stop],start=start):
  path=R/'r2'/(hashlib.sha256(sid.encode()).hexdigest()+'.json');b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==prov[str(path)];d=json.loads(b);c=cp[sid];T=c['T'];z={(x['A'],x['B'],x['tau']):x for x in d['candidates']}
  fig,axes=plt.subplots(4,2,figsize=(11,12),sharex=True,sharey=True)
  for ax,a in zip(axes.flat,A):
   for bi,bb in enumerate(A):
    if bb!=a:ax.plot(range(1,T),[z[a,bb,t]['score'] for t in range(1,T)],marker='.',lw=.85,color=plt.cm.tab10(bi),label=bb)
   ax.axhline(c['native_score'],color='black',ls='--',lw=.8);ax.set_title('Initial action '+a);ax.set_ylim(-.05,1.05);ax.set_ylabel('Reward');ax.set_xlabel('Switch boundary tau');ax.legend(ncol=3,fontsize=6)
  axes.flat[-1].axis('off');fig.suptitle(f'{sid} | Native={c["native_score"]:.3f}; R2={d["r2_score"]:.3f}\nAll 42 ordered action pairs; dashed = Native; same color = final action');fig.tight_layout(rect=[0,0,1,.95]);pdf.savefig(fig);plt.close(fig)
  if (i+1)%25==0:print('CPU all-pair curve pages',i+1,flush=True)
print(f'Complete pages {start+1} through {stop}',flush=True)
