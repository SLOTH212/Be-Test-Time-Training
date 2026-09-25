import pathlib,sys,json,hashlib,os
R=pathlib.Path('/path/to/ttt');S=R/'src/qwen3_4b_downstream_pipeline_v1';W=R/'work/qwen3_4b_fixed8_gpu45_v2';sys.path.insert(0,str(S));import contracts as ct
ct.source_guard();cfg=W/'fixed_config.json';a=ct.read(cfg);orig=ct.read(ct.config_path('fixed'));assert {k:v for k,v in a.items() if k not in ['physical_gpu_allowlist','run_root']}=={k:v for k,v in orig.items() if k not in ['physical_gpu_allowlist','run_root']}
def replace_exact(s,old,new):
 assert s.count(old)==1,(old,s.count(old))
 return s.replace(old,new)
# Preserve original source checks; resource-only code overlay is separately bound.
source=(S/'contracts.py').read_text();start=source.index('def validate_config(');end=source.index('\ndef verify_runtime',start);fragment=source[start:end];fragment=replace_exact(fragment,'physical_gpu_allowlist=[2,3]','physical_gpu_allowlist=[4,5]');exec(compile(fragment,str(S/'contracts.py'),'exec'),ct.__dict__)
old_path=ct.config_path;ct.config_path=lambda phase:cfg if phase=='fixed' else old_path(phase)
worker='--job' in sys.argv
p=S/('formal_worker.py' if worker else 'formal_control.py');source=p.read_text()
if worker:
 source=replace_exact(source,"os.environ.get('CUDA_VISIBLE_DEVICES')=='2,3'","os.environ.get('CUDA_VISIBLE_DEVICES')=='4,5'")
 source=replace_exact(source,"['fb7ba9a8-ce12-65a4-a9cf-5b5b7506b0ed','351e23d4-250b-6a70-2c66-d2fb0221a25a']","['ae0edae4-c3bc-cfb2-4b15-f8cdd888bc25','25896d0f-c552-9793-5c7a-ced116606c44']")
 source=replace_exact(source,'physical_gpu=args.worker_id+2','physical_gpu=args.worker_id+4')
else:
 source=replace_exact(source,'for gpu in [2,3]:','for gpu in [4,5]:')
 source=replace_exact(source,"CUDA_VISIBLE_DEVICES='2,3'","CUDA_VISIBLE_DEVICES='4,5'")
 source=replace_exact(source,"str(ct.S/'formal_worker.py')",repr(str(W/'entry.py')))
if worker:
 source=replace_exact(source,"os.environ.get('CUDA_VISIBLE_DEVICES')=='4,5' and args.worker_id in [0,1]","os.environ.get('CUDA_VISIBLE_DEVICES')=='0,2,4,5' and args.worker_id in [0,1,2,3]")
 source=replace_exact(source,"['ae0edae4-c3bc-cfb2-4b15-f8cdd888bc25','25896d0f-c552-9793-5c7a-ced116606c44']","['c7982600-dea0-9d54-16b8-76db4288663e','fb7ba9a8-ce12-65a4-a9cf-5b5b7506b0ed','ae0edae4-c3bc-cfb2-4b15-f8cdd888bc25','25896d0f-c552-9793-5c7a-ced116606c44']")
 source=replace_exact(source,"owned=j['sample_ids'][args.worker_id::2]","owned=j['sample_ids'][args.worker_id::4]")
 source=replace_exact(source,"workspace=root/'workers'/str(args.worker_id)","workspace=root/'workers'/('scale4_'+str(args.worker_id))")
 source=replace_exact(source,'physical_gpu=args.worker_id+4','physical_gpu=[0,2,4,5][args.worker_id]')
else:raise RuntimeError('SCALE4_WORKER_ONLY')

namespace={'__name__':'gpu45_fixed_resource_overlay','__file__':str(p)};exec(compile(source,str(p),'exec'),namespace)
if worker:namespace['main']()
else:
 from worker import atomic
 receipt={'original_source_manifest_sha256':ct.source_guard(),'original_config_sha256':ct.sha(old_path('fixed')),'resource_config_sha256':ct.sha(cfg),'entry_sha256':ct.sha(__file__),'physical_gpu_allowlist':[4,5],'model_parent_authority_sha256':a['model_parent_authority_sha256'],'scientific_config_unchanged':True,'fixed_only':True,'continuous_monitoring':False}
 rp=W/'RESOURCE_AUTHORITY.json'
 if rp.exists():assert ct.read(rp)==receipt
 else:atomic(rp,receipt)
 sys.argv=[str(p),'launch','--phase','fixed','--parent-authority',a['model_parent_authority'],'--run-root',a['run_root'],'--execute'];namespace['main']()
 sys.argv=[str(p),'finalize','--phase','fixed','--run-root',a['run_root']];namespace['main']()
