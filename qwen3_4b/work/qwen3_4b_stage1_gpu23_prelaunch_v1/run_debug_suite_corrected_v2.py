import os,sys,time,json,pathlib,subprocess,signal,threading,csv,hashlib
root=pathlib.Path("/path/to/ttt")
audit=root/"audits/qwen3_4b_stage1_gpu23_prelaunch_corrected_v2"
audit.mkdir(exist_ok=True)
work=root/"work/qwen3_4b_stage1_gpu23_prelaunch_v1"
runs=root/"runs/qwen3_4b_gpu23_prelaunch_corrected_v2"
worker=root/"src/training_runtime/code/workers/distributed_train_worker.py"
config=work/"real_debug_v2.yaml"
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES="2,3",WORLD_SIZE="2",PYTHONDONTWRITEBYTECODE="1",TMPDIR=str(root/"tmp"),OMP_NUM_THREADS="4",TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
assert env["CUDA_VISIBLE_DEVICES"]=="2,3"
phase="START";quit_log=threading.Event()
def write(name,obj):
 p=audit/name; t=p.with_suffix(".partial");t.write_text(json.dumps(obj,indent=2)+"\n");t.replace(p)
def monitor():
 prior=None
 while not quit_log.is_set():
  v=[int(x) for x in pathlib.Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
  total=sum(v[:8]);idle=v[3]+v[4]
  cpu=None if prior is None else 100*(1-(idle-prior[1])/max(1,total-prior[0]))
  prior=(total,idle)
  row={"time":time.time(),"phase":phase,"cpu_percent":cpu,"cpu_count":os.cpu_count()}
  for line in pathlib.Path("/proc/diskstats").read_text().splitlines():
   d=line.split()
   if d[2]=="sdc":
    row["disk"]={"device":"sdc","read_bytes":int(d[5])*512,"write_bytes":int(d[9])*512,"read_time":int(d[6]),"write_time":int(d[10])}
    break
  try:
   cmd=["nvidia-smi","-i","2,3","--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw,clocks.sm,pstate","--format=csv,noheader,nounits"]
   r=subprocess.run(cmd,capture_output=True,text=True,timeout=8)
   row["gpus"]=list(csv.reader(r.stdout.splitlines()));row["nvidia_smi_rc"]=r.returncode
  except Exception as e:row["error"]=repr(e)
  with (audit/"GPU23_UTILIZATION_IO.jsonl").open("a") as f:f.write(json.dumps(row)+"\n")
  quit_log.wait(5)
def execute(name,command,expected_failure=False,timeout=3600):
 global phase
 phase=name;started=time.time();log=audit/(name+".log")
 with log.open("x") as f:
  p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=root,env=env,start_new_session=True)
  write("DEBUG_SUITE_STATUS.json",{"phase":name,"pid":p.pid,"started":started,"command":command,"formal_training_started":False})
  try:rc=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   raise RuntimeError("BOUNDED_DEBUG_JOB_TIMEOUT "+name)
 receipt={"phase":name,"exit_code":rc,"seconds":time.time()-started,"expected_failure":expected_failure,"command":command}
 write(name+"_EXIT.json",receipt)
 if (expected_failure and rc==0) or (not expected_failure and rc!=0):raise RuntimeError("DEBUG_JOB_EXIT_MISMATCH "+name)
 if expected_failure:
  text=log.read_text(errors="replace")
  if "CONTROLLED_DEBUG_SOFTWARE_FAILURE" not in text:raise RuntimeError("UNEXPECTED_FAILURE_CAUSE")
 return receipt
def train(name,directory,maximum,extra=(),cfg=config,expected_failure=False):
 cmd=[str(root/"envs/ttt_runtime_v1/bin/torchrun"),"--standalone","--nproc-per-node=2",str(worker),"--stage","1","--config",str(cfg),"--output-dir",str(runs/directory),"--max-records",str(maximum),*extra]
 return execute(name,cmd,expected_failure)
def compare(name,a,b):
 original=(root/"src/training_runtime/code/tests/hit_dcp_numeric_compare.py").read_text()
 import yaml
 cfg=yaml.safe_load(config.read_text())
 identity={"config_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),**cfg["input_identity"]}
 original=original.replace('ROOT = Path(__file__).resolve().parents[2]',f'ROOT = Path({str(root/"src/training_runtime")!r})')
 original=original.replace('A_ROOT = ROOT / "runs/l4_uninterrupted_v1/checkpoints"',f'A_ROOT = Path({str(a)!r})')
 original=original.replace('B_ROOT = ROOT / "runs/l4_exact_a_step1_snapshot/checkpoints"',f'B_ROOT = Path({str(b)!r})')
 original=original.replace('AUDIT = ROOT / "audits/hit_l3_l4_v1/L4_NUMERIC_STATE_COMPARE.json"',f'AUDIT = Path({str(audit/(name+".json"))!r})')
 original=original.replace('BATCH = {"global_batch_size": 2, "micro_batch_size": 1, "gradient_accumulation": 1}',f'BATCH = {dict(global_batch_size=2,micro_batch_size=1,gradient_accumulation=1,identity=identity)!r}')
 path=work/(name+".py");path.write_text(original)
 execute(name,[str(root/"envs/ttt_runtime_v1/bin/torchrun"),"--standalone","--nproc-per-node=2",str(path)])
try:
 assert (root/"runs/qwen3_4b_gpu23_prelaunch_v1/smoke_A/complete.json").exists(),"INITIAL_SMOKE_NOT_COMPLETE"
 assert not (audit/"SUITE_COMPLETED.json").exists()
 t=threading.Thread(target=monitor,daemon=True);t.start()
 train("THROUGHPUT_25","throughput_A",50)
 aroot=runs/"throughput_A/checkpoints";slot=aroot/"slot_B";manifest=json.loads((slot/"manifest.json").read_text())
 assert manifest["progress"]["update_step"]==5
 anchor=runs/"uninterrupted_step5_anchor/checkpoints";anchor.mkdir(parents=True,exist_ok=False)
 (anchor/"slot_B").symlink_to(slot,target_is_directory=True)
 (anchor/"latest.json").write_text(json.dumps({"slot":"slot_B","progress":manifest["progress"],"manifest_sha256":hashlib.sha256((slot/"manifest.json").read_bytes()).hexdigest()}))
 train("RESUME_B_BEFORE","resume_B",4)
 train("RESUME_B_AFTER","resume_B",6,["--resume-from",str(runs/"resume_B/checkpoints")])
 compare("REAL_RESUME_NUMERIC_PARITY",anchor,runs/"resume_B/checkpoints")
 train("FAILURE_C_INJECT","failure_C",4,["--debug-fail-after-save-step","2"],expected_failure=True)
 assert not (runs/"failure_C/complete.json").exists(),"FALSE_COMPLETE"
 assert (runs/"failure_C/checkpoints/latest.json").exists(),"VALID_DCP_MISSING_AFTER_FAILURE"
 train("FAILURE_C_RESUME","failure_C",6,["--resume-from",str(runs/"failure_C/checkpoints")])
 compare("REAL_FAILURE_RECOVERY_NUMERIC_PARITY",anchor,runs/"failure_C/checkpoints")
 train("EDGE_ONE_TOKEN","edge_one_token",2,["--debug-start-record","15220"],cfg=work/"real_edge_debug.yaml")
 train("EDGE_ODD_FINAL","edge_odd_final",1,["--debug-start-record","38334"],cfg=work/"real_edge_debug.yaml")
 write("SUITE_COMPLETED.json",{"status":"PASS","formal_training_started":False,"finished":time.time()})
 write("DEBUG_SUITE_STATUS.json",{"phase":"COMPLETE","formal_training_started":False})
except Exception as e:
 write("DEBUG_SUITE_STATUS.json",{"phase":"FAILED","error":repr(e),"formal_training_started":False})
 raise
finally:
 quit_log.set()
