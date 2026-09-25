import pathlib,json,os,time,subprocess,sys
R=pathlib.Path("/path/to/ttt");W=R/"work/qwen3_4b_stage1_gpu23_prelaunch_v1";A=R/"audits/qwen3_4b_stage1_gpu23_prelaunch_determinism_v4"
assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
driver=3287424
def identity(pid):
 try:
  p=pathlib.Path("/proc")/str(pid);stat=(p/"stat").read_text().rsplit(")",1)[1].split()
  return (stat[19],(p/"cmdline").read_bytes())
 except FileNotFoundError:return None
initial=identity(driver)
assert initial and b"run_deterministic_completion_v4.py" in initial[1],"EXPECTED_DRIVER_NOT_LIVE"
with (A/"CPU_FINALIZATION_STARTED.json").open("x") as f:json.dump({"pid":os.getpid(),"driver_pid":driver,"driver_start_ticks":initial[0],"started":time.time()},f)
def state(phase,**extra):
 p=A/"CPU_FINALIZATION_STATUS.json";tmp=p.with_suffix(".partial");tmp.write_text(json.dumps({"phase":phase,"formal_training_started":False,**extra},indent=2)+"\n");tmp.replace(p)
try:
 state("WAITING_FOR_EXISTING_GPU_SUITE_EXIT",driver_pid=driver,pid=os.getpid())
 deadline=time.monotonic()+4*3600
 while identity(driver)==initial:
  if time.monotonic()>deadline:raise RuntimeError("BOUNDED_WAIT_FOR_DRIVER_EXCEEDED")
  time.sleep(5)
 assert json.loads((A/"SUITE_COMPLETED.json").read_text())["status"]=="PASS","GPU_SUITE_NOT_PASS"
 for name in ["analyze_deterministic_throughput_v4.py","final_real_evidence_v4.py","freeze_for_final_audit_v4.py"]:
  state(name,pid=os.getpid())
  result=subprocess.run([sys.executable,"-B",str(W/name)],cwd=R,env=os.environ.copy(),timeout=1800)
  assert result.returncode==0,("CPU_FINALIZATION_FAILED",name,result.returncode)
 state("READY_FOR_INDEPENDENT_REVIEW",pid=os.getpid())
except Exception as exc:
 state("FAILED",error=repr(exc),pid=os.getpid());raise
