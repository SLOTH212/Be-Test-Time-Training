#!/usr/bin/env python3
"""Fail-closed formal launcher/status. Creation never launches training."""
import argparse,csv,fcntl,hashlib,json,os,pathlib,subprocess,sys,time,pwd
ROOT=pathlib.Path("/path/to/ttt")
NAME="qwen3_4b_stage1_1b_32k_gpu23_formal_v1"
CFG=ROOT/"configs"/(NAME+".yaml")
AUTH=ROOT/"provenance/QWEN3_4B_STAGE1_1B_32K_GPU23_PRELAUNCH_AUTHORITY_V1.json"
OUT=ROOT/"runs"/NAME
PID=ROOT/"runs"/(NAME+".process.json")
WORKER=ROOT/"src/training_runtime/code/workers/distributed_train_worker.py"
def digest(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def owned(p):
 p=pathlib.Path(p).resolve()
 if not p.is_relative_to(ROOT.resolve()):raise RuntimeError("OUTPUT_PATH_OUTSIDE_AUTHORIZED_ROOT")
 return p
def atomic(p,obj):
 p=owned(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(".partial")
 t.write_text(json.dumps(obj,indent=2)+"\n");t.replace(p)
def query(fields,apps=False):
 q="--query-compute-apps=" if apps else "--query-gpu="
 p=subprocess.run(["nvidia-smi","-i","2,3",q+fields,"--format=csv,noheader,nounits"],capture_output=True,text=True,check=True,timeout=15)
 return [[x.strip() for x in r] for r in csv.reader(p.stdout.splitlines()) if r]
def process_identity(pid):
 try:
  p=pathlib.Path('/proc')/str(int(pid))
  fields=(p/'stat').read_text().rsplit(')',1)[1].split()
  return {'start_ticks':int(fields[19]),'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace').strip()}
 except (FileNotFoundError,ProcessLookupError,ValueError,TypeError):return None
def alive(pid,expected):
 current=process_identity(pid)
 return bool(current and expected and current==expected)
def guard():
 import yaml
 if os.uname().nodename!="amax" or pwd.getpwuid(os.getuid()).pw_name!="USER" or os.environ.get("HOME")!="/home/USER":raise RuntimeError("HOST_ACCOUNT_IDENTITY_MISMATCH")
 if os.environ.get("CUDA_VISIBLE_DEVICES")!="2,3" or os.environ.get("WORLD_SIZE")!="2":raise RuntimeError("GPU_POLICY_VIOLATION")
 if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8" or os.environ.get("PYTHONHASHSEED")!="42":raise RuntimeError("DETERMINISTIC_ENVIRONMENT_REQUIRED_BEFORE_PROCESS_START")
 subprocess.run([str(ROOT/"bin/check_shared_assets.sh")],check=True)
 auth=json.loads(AUTH.read_text())
 if auth.get("QWEN3_4B_STAGE1_PRELAUNCH_STATUS") not in ("PASS","PASS_WITH_WARNING") or auth.get("SAFE_TO_LAUNCH_QWEN3_4B_STAGE1_1B_32K_GPU23")!="YES":raise RuntimeError("PRELAUNCH_AUTHORITY_NOT_PASS")
 checksum=AUTH.with_suffix(".sha256").read_text().split()[0]
 if digest(AUTH)!=checksum:raise RuntimeError("PRELAUNCH_AUTHORITY_SHA_MISMATCH")
 for path,expected in auth["file_sha256"].items():
  if digest(path)!=expected:raise RuntimeError("PRELAUNCH_FILE_IDENTITY_MISMATCH "+path)
 cfg=yaml.safe_load(CFG.read_text())
 wrapper=owned(cfg["execution_wrapper"])
 if wrapper!=ROOT/"bin/qwen3_4b_stage1_deterministic_wrapper_v1.py" or digest(wrapper)!=cfg["deterministic_runtime"]["wrapper_sha256"]:raise RuntimeError("DETERMINISTIC_WRAPPER_IDENTITY_MISMATCH")
 import importlib.metadata
 expected_env=json.loads(pathlib.Path(cfg["environment_identity_receipt"]).read_text())
 if sys.version!=expected_env["python"] or str(pathlib.Path(sys.executable).resolve())!=expected_env["executable"]:raise RuntimeError("PYTHON_ENVIRONMENT_IDENTITY_MISMATCH")
 for name,version in expected_env["packages"].items():
  if importlib.metadata.version(name)!=version:raise RuntimeError("PACKAGE_ENVIRONMENT_IDENTITY_MISMATCH "+name)
 if cfg["world_size"]!=2 or cfg["physical_gpu_allowlist"]!="2,3" or cfg["global_batch_size"]!=cfg["micro_batch_size"]*2*cfg["gradient_accumulation"]:raise RuntimeError("FORMAL_BATCH_GPU_EQUATION")
 for key in ["output_root","stage1_model_only_artifact"]:owned(cfg[key])
 source=json.loads(pathlib.Path(cfg["runtime_source_manifest"]).read_text())
 for rel,expected in source.items():
  if digest(ROOT/"src/training_runtime"/rel)!=expected:raise RuntimeError("RUNTIME_SOURCE_HASH_MISMATCH "+rel)
 model_receipt=json.loads(pathlib.Path(cfg["model_identity_receipt"]).read_text())
 model=pathlib.Path(cfg["model_base"])
 for rel,row in model_receipt["weights"].items():
  if digest(model/rel)!=row["sha256"]:raise RuntimeError("MODEL_WEIGHT_IDENTITY_MISMATCH")
 ds=pathlib.Path(cfg["train_data"]).parent
 entries=[json.loads(x) for x in (ds/"manifests/FINAL_DATA_FILE_HASHES.jsonl").read_text().splitlines()]
 for row in entries:
  p=(ds/row["path"]).resolve()
  if not p.is_relative_to(ds.resolve()) or digest(p)!=row["sha256"]:raise RuntimeError("DATA_SHARD_IDENTITY_MISMATCH")
 gpus=query("index,uuid,memory.free,driver_version")
 if [int(r[0]) for r in gpus]!=[2,3]:raise RuntimeError("PHYSICAL_GPU_SET_MISMATCH")
 for r in gpus:
  i=int(r[0])
  if r[1]!=cfg["gpu_uuids"][i] or float(r[2])<cfg["minimum_free_memory_mib"][i] or r[3]!=expected_env["driver_version"]:raise RuntimeError("GPU_UNAVAILABLE_FAIL_CLOSED")
 allowed={(r["uuid"],str(r["pid"]),r["process_name"]) for r in cfg.get("authorized_existing_gpu_processes",[])}
 actual=query("gpu_uuid,pid,process_name,used_gpu_memory",True)
 if any(tuple(r[:3]) not in allowed for r in actual):raise RuntimeError("UNAUTHORIZED_CONCURRENT_GPU_WORKLOAD")
 return cfg,checksum
def supervise(lockfd):
 if lockfd is None:raise RuntimeError("SUPERVISOR_INHERITED_LOCK_REQUIRED")
 st=os.fstat(lockfd);expected=(ROOT/"runs"/(NAME+".lock")).stat()
 if (st.st_dev,st.st_ino)!=(expected.st_dev,expected.st_ino) or st.st_uid!=os.getuid():raise RuntimeError("SUPERVISOR_LOCK_IDENTITY_MISMATCH")
 fcntl.flock(lockfd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 cfg,authority_sha=guard()
 started=time.time()
 command=[str(ROOT/"envs/ttt_runtime_v1/bin/torchrun"),"--standalone","--nproc-per-node=2",str(owned(cfg["execution_wrapper"])),"--stage","1","--config",str(CFG),"--output-dir",str(OUT)]
 with (OUT/"launcher.log").open("x") as log:
  p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=os.environ.copy())
  atomic(PID,{"supervisor_pid":os.getpid(),"torchrun_pid":p.pid,"torchrun_identity":process_identity(p.pid),"started":started,"command":command,"FORMAL_STAGE1_TRAINING_STARTED":True})
  rc=p.wait()
 complete=OUT/"complete.json"
 valid=rc==0 and complete.exists()
 if valid:
  import yaml
  cfg=yaml.safe_load(CFG.read_text());record=json.loads(complete.read_text())
  valid=record["record_cursor"]==cfg["record_cursor_final"] and record["cumulative_tokens"]==cfg["train_tokens"]
 atomic(OUT/"launcher_exit.json",{"exit_code":rc,"complete_verified":valid,"ended":time.time(),"elapsed_seconds":time.time()-started})
 os.close(lockfd)
 return 0 if valid else 1
def launch():
 cfg,authority_sha=guard()
 fd=os.open(owned(ROOT/"runs"/(NAME+".lock")),os.O_CREAT|os.O_RDWR,0o600)
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if OUT.exists():raise RuntimeError("FORMAL_OUTPUT_EXISTS_EXPLICIT_RESUME_REQUIRED")
 owned(OUT).mkdir(parents=True)
 with (OUT/"supervisor.log").open("x") as log:
  p=subprocess.Popen([sys.executable,__file__,"supervise","--lock-fd",str(fd)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT,env=os.environ.copy(),start_new_session=True,pass_fds=(fd,))
 os.close(fd)
 print(json.dumps({"detached_supervisor_pid":p.pid,"output":str(OUT),"prelaunch_authority_sha256":authority_sha}))
def status():
 state=json.loads(PID.read_text()) if PID.exists() else {}
 result={"formal_training_started":bool(state.get("FORMAL_STAGE1_TRAINING_STARTED",False)),"process":state,"process_alive":alive(state.get("torchrun_pid"),state.get("torchrun_identity")),"output":str(OUT)}
 f=OUT/"training.jsonl";last=None
 if f.exists():
  with f.open("rb") as h:
   h.seek(max(0,f.stat().st_size-(8<<20)))
   for line in reversed(h.read().splitlines()):
    try:last=json.loads(line);break
    except (ValueError,UnicodeDecodeError):pass
 if last:
  result.update(input_tokens=last["cumulative_tokens"],optimizer_step=last["update_step"],loss=last["global_mean_loss"])
  dt=last.get("iteration_seconds_excluding_checkpoint",0)
  tokens=sum(r.get("local_tokens",0) for r in last["rank_runtime"])
  result["instant_step_tokens_per_sec"]=tokens/dt if dt else None
 if CFG.exists():
  import yaml
  cfg=yaml.safe_load(CFG.read_text());result.update(dataset_logical_budget=cfg["dataset_logical_tokens"],train_logical_budget=cfg["train_logical_budget_tokens"],train_input_token_target=cfg["train_tokens"])
  rate=result.get("instant_step_tokens_per_sec")
  result["ETA_seconds"]=(cfg["train_tokens"]-result.get("input_tokens",0))/rate if rate else None
 latest=OUT/"checkpoints/latest.json"
 result["latest_checkpoint"]=json.loads(latest.read_text()) if latest.exists() else None
 result["elapsed_seconds"]=time.time()-state["started"] if "started" in state else None
 try:result["GPU2_GPU3"]=query("index,utilization.gpu,memory.used,power.draw")
 except Exception as e:result["gpu_query_error"]=str(e)
 exitfile=OUT/"launcher_exit.json";result["exit"]=json.loads(exitfile.read_text()) if exitfile.exists() else None
 if result["exit"] and "started" in state:result["elapsed_seconds"]=result["exit"]["ended"]-state["started"]
 result["errors"]=[]
 for name in ["launcher.log","supervisor.log"]:
  log=OUT/name
  if log.exists():
   with log.open("rb") as f:
    f.seek(max(0,log.stat().st_size-(1<<16)));lines=f.read().decode(errors="replace").splitlines()
   result["errors"].extend([name+": "+line for line in lines if any(x in line for x in ["Error","Traceback","FAILED","NCCL WARN","CUDA out of memory"])][-20:])
 startup=OUT/"supervisor_startup_failure.json"
 if startup.exists():result["supervisor_startup_failure"]=json.loads(startup.read_text())
 print(json.dumps(result,indent=2))
if __name__=="__main__":
 parser=argparse.ArgumentParser();parser.add_argument("action",choices=["launch","status","verify","supervise"]);parser.add_argument("--lock-fd",type=int)
 args=parser.parse_args()
 if args.action=="status":status()
 elif args.action=="verify":print(json.dumps({"guard":"PASS","authority_sha256":guard()[1]}))
 elif args.action=="launch":launch()
 else:
  try:sys.exit(supervise(args.lock_fd))
  except Exception as exc:
   if OUT.exists():atomic(OUT/"supervisor_startup_failure.json",{"error":repr(exc),"ended":time.time(),"formal_start_observed":PID.exists()})
   raise
