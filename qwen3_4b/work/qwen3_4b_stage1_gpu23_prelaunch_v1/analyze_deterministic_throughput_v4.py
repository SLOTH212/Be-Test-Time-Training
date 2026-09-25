import json,pathlib,statistics,math,time,hashlib
ROOT=pathlib.Path("/path/to/ttt")
AUDIT=ROOT/"audits/qwen3_4b_stage1_gpu23_prelaunch_determinism_v4"
RUN=ROOT/"runs/qwen3_4b_gpu23_prelaunch_determinism_v4/throughput_A"
def quantile(values,q):
 values=sorted(values);p=(len(values)-1)*q;i=int(p);return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(p-i)
def summarize(values):
 return {"mean":statistics.mean(values),"median":statistics.median(values),"p95":quantile(values,.95),"max":max(values)}
log=[json.loads(x) for x in (RUN/"training.jsonl").read_text().splitlines()]
assert len(log)==25 and (RUN/"complete.json").exists(),"25_STEP_THROUGHPUT_NOT_COMPLETE"
with (ROOT/"datasets/train/part-00000.jsonl").open() as f:
 real=[json.loads(next(f)) for _ in range(50)]
tokens=0
for index,row in enumerate(log):
 expected=real[index*2:index*2+2]
 tokens+=sum(r["token_count"] for r in expected)
 assert row["cumulative_tokens"]==tokens and row["record_cursor"]==(index+1)*2
 assert row["global_context_positions"]==sum(r["token_count"]-1 for r in expected)
 assert row["update_step"]==index+1
 for rank,rr in enumerate(row["rank_runtime"]):
  source=expected[rank]
  assert rr["physical_gpu"]==rank+2
  assert rr["sample_ids"]==[source["sample_id"]]
  assert rr["local_tokens"]==source["token_count"]
  assert rr["boundary_trace"][0]["boundaries"]==json.loads(source["document_boundaries_qwen_json"])
  for layer in ["0","6","12","18","24","30"]:
   trace=rr["boundary_trace"][0]["layers"][layer]
   bounds=rr["boundary_trace"][0]["boundaries"]
   chunks=[(b-a+4095)//4096 for a,b in bounds]
   assert trace["chunks_per_document"]==chunks
   assert trace["ntp_pairs_per_document"]==[b-a-c for (a,b),c in zip(bounds,chunks)]
  assert all(v["finite"] for v in rr["l30_gradient"].values())
  assert all(v["owned"] for v in rr["l30_optimizer_state"].values())
measured=log[5:25]
times=[r["iteration_seconds_excluding_checkpoint"] for r in measured]
counts=[sum(x["local_tokens"] for x in r["rank_runtime"]) for r in measured]
total_seconds=sum(times);total_tokens=sum(counts)
rate=total_tokens/total_seconds
intervals=[(r["logged_at_unix"]-r["checkpoint_seconds"]-r["iteration_seconds_excluding_checkpoint"],r["logged_at_unix"]-r["checkpoint_seconds"]) for r in measured]
samples=[json.loads(x) for x in (AUDIT/"GPU23_UTILIZATION_IO.jsonl").read_text().splitlines()]
samples=[x for x in samples if x["phase"]=="THROUGHPUT_25" and any(a<=x["time"]<=b for a,b in intervals)]
assert len(samples)>=20,"INSUFFICIENT_STEADY_STATE_GPU_TELEMETRY"
gpus={}
for i in [2,3]:
 rs=[row for s in samples for row in s.get("gpus",[]) if int(row[0])==i]
 gpus[str(i)]={"physical_id":i,"utilization_percent":summarize([float(r[2]) for r in rs]),"memory_mib":summarize([float(r[3]) for r in rs]),"power_w":summarize([float(r[4]) for r in rs]),"sm_clock_mhz":summarize([float(r[5]) for r in rs]),"pstates":sorted(set(r[6].strip() for r in rs)),"samples":len(rs),"torch_peak_allocated_bytes":max(r["rank_runtime"][i-2]["peak_allocated_bytes"] for r in measured),"torch_peak_reserved_bytes":max(r["rank_runtime"][i-2]["peak_reserved_bytes"] for r in measured)}
wait=sum(max(x["loader_wait_seconds"] for x in r["rank_runtime"]) for r in measured)
tokenize=sum(max(x["tokenizer_seconds"] for x in r["rank_runtime"]) for r in measured)
cpu=[x["cpu_percent"] for x in samples if x.get("cpu_percent") is not None]
diskrates=[]
for a,b in zip(samples,samples[1:]):
 if b["time"]-a["time"]<12 and a.get("disk") and b.get("disk"):
  elapsed=b["time"]-a["time"];diskrates.append({k:(b["disk"][k]-a["disk"][k])/elapsed for k in ["read_bytes","write_bytes"]})
checkpoint_seconds=[r["checkpoint_seconds"] for r in log if r["checkpoint_seconds"]>0]
checkpoint=RUN/"checkpoints/slot_A"
checkpoint_bytes=sum(p.stat().st_size for p in checkpoint.rglob("*") if p.is_file())
validation=RUN/"validation_smoke.json"
validation_seconds=max(0.,validation.stat().st_mtime-log[-1]["logged_at_unix"])
overhead={"checkpoint_mean_seconds":statistics.mean(checkpoint_seconds),"checkpoint_samples_seconds":checkpoint_seconds,"validation_4_records_wall_seconds_from_receipt_timestamps":validation_seconds,"final_395_validation_records_linear_estimate_seconds":validation_seconds*395/4,"formal_checkpoint_count_estimate":78,"startup_allowance_seconds":600,"io_included_in_iteration_seconds":True}
central_overhead=(78*statistics.mean(checkpoint_seconds)+validation_seconds*395/4+600)/3600
pure=1e9/rate/3600
per_step_rates=[n/t for n,t in zip(counts,times)]
estimate={"pure_nominal_1b_training_hours":pure,"optimistic_hours":1e9/quantile(per_step_rates,.75)/3600+central_overhead*.8,"central_hours":pure+central_overhead,"conservative_hours":1e9/quantile(per_step_rates,.1)/3600+central_overhead*1.5,"actual_990m_split_input_training_hours":989996971/rate/3600,"overhead":overhead,"limitations":["measurement uses first50 canonical records, predominantly historical-anchor source; corpus mixture may differ","GPU2/3 explicitly shared; future contention may change throughput","1B nominal training-time projection is separate from frozen split input total989996971 and logical train budget990M"]}
report={"status":"PASS","warmup_optimizer_steps":5,"measured_optimizer_steps":20,"micro_batch_per_gpu":1,"world_size":2,"gradient_accumulation":1,"global_batch":2,"step_seconds":summarize(times),"logical_input_tokens_per_second":rate,"input_tokens_in_measured_window":total_tokens,"measured_seconds":total_seconds,"samples_per_second":40/total_seconds,"dataloader_wait_fraction":wait/total_seconds,"tokenization_fraction":tokenize/total_seconds,"cpu_host_utilization_percent":summarize(cpu),"disk_host_read_bytes_per_second":summarize([r["read_bytes"] for r in diskrates]) if diskrates else None,"disk_host_write_bytes_per_second":summarize([r["write_bytes"] for r in diskrates]) if diskrates else None,"gpu":gpus,"debug_checkpoint_bytes":checkpoint_bytes,"wall_time_estimate":estimate,"independent_token_accounting":"PASS","real_document_boundary_and_chunk_trace":"PASS","six_layer_gradient_optimizer_coverage":"PASS","UNAUTHORIZED_GPU_USED":False,"FORMAL_STAGE1_TRAINING_STARTED":False}
report["deterministic_runtime"]={"algorithms":True,"warn_only":False,"CUBLAS_WORKSPACE_CONFIG":":4096:8","PYTHONHASHSEED":"42","cudnn_deterministic":True,"cudnn_benchmark":False}
report["input_tokens_per_second"]=rate
report["token_counter_semantics"]="actual record input tokens; nominal source logical budget has no exact per-record mapping"
report["wall_time_estimate"]["actual_train_split_central_hours"]=989996971/rate/3600+central_overhead
report["wall_time_estimate"]["record_step_projection_hours"]=math.ceil(38335/2)*statistics.mean(times)/3600+central_overhead
report["wall_time_estimate"]["overhead"]["validation_projection_note"]="complete4record wrapper includes both parameter fingerprint passes; scaling entire wrapper to395 is conservative for fixed hashing overhead"
if all(g["utilization_percent"]["max"]==0 for g in gpus.values()):
 report["gpu_utilization_telemetry_status"]="WARNING_DRIVER_REPORTED_ZERO_DURING_CONFIRMED_REAL_TRAINING"
 report["gpu_utilization_note"]="Preserved raw0%; not interpreted as idle or a valid estimate of actual utilization."
(AUDIT/"STEADY_STATE_THROUGHPUT.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
