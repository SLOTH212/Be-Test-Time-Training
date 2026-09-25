"""Explicit user-authorized repair; preserve the failed attempt and scientific identities."""
import ast
import shutil
from core import *

c=config();root=Path(c['state_root']);run=Path(c['stage1']['run_root'])
with lock(root/'PIPELINE.lock'), lock(root/'STAGE1.lock'):
 require(not trainer_snapshot(1,c)['active'],'TRAINER_STILL_ACTIVE')
 require(read(root/'PIPELINE_STATE.json')['state']=='BLOCKED','EXPECTED_FAILED_RUN')
 backup=root/'repair_v2_original'
 require(not backup.exists(),'REPAIR_ALREADY_INSTALLED')
 source()
 backup.mkdir()
 for p in [SOURCE,SOURCE.with_suffix('.sha256'),W/'entry.py',W/'pipeline.py',root/'EXECUTION_AUTHORIZED.json']:
  shutil.copy2(p,backup/p.name)
 s=(W/'entry.py').read_text();s=s.replace("g['main']()", "from checkpoint_io_v2 import install as install_checkpoint_io\ninstall_checkpoint_io(g, cp)\ng['main']()")
 ast.parse(s);(W/'entry.py').write_text(s)
 s=(W/'pipeline.py').read_text()
 s=s.replace('def trainer(stage,c,parent):','def trainer(stage,c,parent,resume=False):')
 s=s.replace("require(not run.exists(),'DUPLICATE_TRAINER_ROOT')", "require(run.exists() if resume else not run.exists(),'TRAINER_ROOT_CONTRACT')")
 s=s.replace("  run.mkdir();receipt=", "  if not resume:run.mkdir()\n  receipt=")
 s=s.replace("   if stage==2:\n", "   if resume:command+=['--resume-from',str(run/'checkpoints')]\n   if stage==2:\n",1)
 s=s.replace("stderr=subprocess.STDOUT,env=env(),cwd=R", "stderr=subprocess.STDOUT,env={**env(),**({'LLAMA_RESUME_VERIFY_SAVE':'1'} if resume else {})},cwd=R",1)
 s=s.replace("'finalize','prune']", "'finalize','prune','resume']")
 s=s.replace("p.add_argument('--parent-authority');", "p.add_argument('--parent-authority');p.add_argument('--resume',action='store_true');")
 s=s.replace('trainer(a.stage,c,a.parent_authority);return','trainer(a.stage,c,a.parent_authority,a.resume);return')
 marker=" if a.mode=='supervise':supervise(c);return"
 s=s.replace(marker, " if a.mode=='resume':\n  require(a.execute,'EXPLICIT_RESUME_EXECUTE_REQUIRED')\n  from resume_v2 import dispatch\n  dispatch(c);return\n"+marker)
 ast.parse(s);(W/'pipeline.py').write_text(s)
 authority=read(SOURCE)
 authority.update(CHANGE_CLASS='CHECKPOINT_IO_SYNCHRONIZATION_AND_EXPLICIT_RESUME',previous_source_authority_sha256=sha(backup/SOURCE.name),repair_scope='Llama-only CPU checkpoint barriers; explicit same-state resume; scientific configs and shared runtime unchanged')
 for name in ['entry.py','pipeline.py','checkpoint_io_v2.py','resume_v2.py']:
  authority['files'][str(W/name)]=sha(W/name)
 atomic(SOURCE,authority);SOURCE.with_suffix('.sha256').write_text(sha(SOURCE)+'  '+SOURCE.name+'\n')
 atomic(root/'EXECUTION_AUTHORIZED.json',{'source_sha256':sha(SOURCE),'time':time.time(),'execute':True,'user_authorization':'Repair, validate, then resume pipeline','previous_authorization_sha256':sha(backup/'EXECUTION_AUTHORIZED.json')})
 source()
 print(json.dumps({'installed':True,'source_sha256':sha(SOURCE),'backup':str(backup)}))
