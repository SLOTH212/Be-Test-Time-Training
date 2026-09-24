import os,sys,subprocess,json
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',NVIDIA_VISIBLE_DEVICES='none',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
P=Path(__file__).resolve().parent
sys.path.insert(0,'/tmp/r2_cpu_plot_deps')
from pypdf import PdfReader,PdfWriter
parts=[]
for start in range(0,170,20):
 subprocess.run([sys.executable,str(P/'all_curves.py'),str(start)],check=True)
 part=P/'figures'/f'curves_part_{start:03d}.pdf';assert len(PdfReader(part).pages)==min(20,170-start);parts.append(part)
w=PdfWriter()
for part in parts:w.append(part)
out=P/'figures/all_170_pair_curves.pdf'
with out.open('wb') as f:w.write(f)
w.close();assert len(PdfReader(out).pages)==170
print('Verified 170 pages / 7140 pair curves',flush=True)
