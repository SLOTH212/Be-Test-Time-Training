# BLOCKED_MAGNITUDE_MATCHING

Traceback (most recent call last):
  File "/home/zonghan/ttt/runs/formal/matched_nonzero_control_v1_20260926/run_control.py", line 165, in <module>
    else:run(phase,worker)
         ^^^^^^^^^^^^^^^^^
  File "/home/zonghan/ttt/runs/formal/matched_nonzero_control_v1_20260926/run_control.py", line 149, in run
    r=process_one(c,mod,ctx,m,tok,rows[c['sample_id']],hist,cfg,smoke=phase=='smoke');put(p,r);done+=1
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/zonghan/ttt/runs/formal/matched_nonzero_control_v1_20260926/run_control.py", line 117, in process_one
    raise RuntimeError('BLOCKED_MAGNITUDE_MATCHING: closest represented norm exceeds 1pct; no formal continuation')
RuntimeError: BLOCKED_MAGNITUDE_MATCHING: closest represented norm exceeds 1pct; no formal continuation

No cohort or seed changes. See magnitude_failure.json if present. Formal execution halted.
