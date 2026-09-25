from pathlib import Path
import json,hashlib,tarfile,time
R=Path('/path/to/ttt');A=R/'audits/v3';W=R/'work/qwen3_4b_stage1_resume_v3';D=A/'analysis'
report=R/'reports/qwen3_4b_stage1_resume_numerical_equivalence_v3_cpu_review_pending_localization.md'
audit=R/'audits/qwen3_4b_stage1_resume_equivalence_v3/CPU_REVIEW_PENDING_LOCALIZATION_AUDIT.json'
obj=json.loads(audit.read_text());assert obj['final_closure'] is False and obj['required_fields']['SAFE_TO_LAUNCH_QWEN3_4B_STAGE1_1B_32K_GPU23']=='NO'
files=[report,audit]+sorted(A.glob('*.json'))+sorted((A/'execution2').glob('*.json'))+[D/n for n in ['ALL_TENSOR_PAIR_METRICS.jsonl','ALL_TENSOR_INFERENCE.json','PAIR_METRIC_MANIFEST.json','PAIR_METRICS_COMPLETE.json','RESAMPLING_DESIGN.json','ALL_TENSOR_PAIR_METRICS_UNOPTIMIZED_ATTEMPT.jsonl','ALL_TENSOR_PAIR_METRICS_SERIAL_CACHED_ATTEMPT.bin']]+sorted(W.glob('*.py'))
files=sorted(set(files));manifest={str(p.relative_to(R)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in files}
mp=R/'reports/qwen3_4b_stage1_resume_v3_cpu_review_bundle_manifest.json';mp.write_text(json.dumps({'created_at':time.time(),'final_closure':False,'raw_pt_files_included':False,'raw_pt_location':'runs/qwen3_4b_stage1_resume_v3/attempt2 on server; 20 snapshots are hash-bound by PAIR_METRIC_MANIFEST.json','files':manifest},indent=2)+'\n')
archive=R/'reports/qwen3_4b_stage1_resume_v3_cpu_review_evidence.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=5) as tf:
 for p in files+[mp]:tf.add(p,arcname=str(p.relative_to(R)),recursive=False)
print(json.dumps({'report':str(report),'audit':str(audit),'archive':str(archive),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,'manifest':str(mp)},indent=2))
