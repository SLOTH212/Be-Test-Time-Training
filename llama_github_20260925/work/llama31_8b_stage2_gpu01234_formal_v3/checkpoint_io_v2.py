"""Llama-only checkpoint I/O synchronization; training collectives unchanged."""
from datetime import timedelta
import json
import os
import time
import torch.distributed as dist


def install(worker_globals, checkpoint_module, timeout_seconds=1800):
    original_initialize = worker_globals['initialize']
    group = None

    def initialize():
        nonlocal group
        context = original_initialize()
        group = dist.new_group(backend='gloo', timeout=timedelta(seconds=timeout_seconds))
        return context

    def io_barrier():
        if dist.is_initialized():
            if group is None:
                raise RuntimeError('CHECKPOINT_IO_GROUP_NOT_INITIALIZED')
            dist.monitored_barrier(group=group, timeout=timedelta(seconds=timeout_seconds), wait_all_ranks=True)

    worker_globals['initialize'] = initialize
    checkpoint_module._barrier = io_barrier
    base = worker_globals['DistributedCheckpointRotation']

    class Rotation(base):
        def save(self, *args, **kwargs):
            started = time.time()
            print(json.dumps({'event': 'CHECKPOINT_IO_BEGIN', 'rank': dist.get_rank(), 'time': started}), flush=True)
            result = super().save(*args, **kwargs)
            print(json.dumps({'event': 'CHECKPOINT_IO_PASS', 'rank': dist.get_rank(), 'seconds': time.time()-started}), flush=True)
            return result

        def load(self, model, optimizer, scheduler=None):
            progress = super().load(model, optimizer, scheduler)
            io_barrier()
            if progress is not None and os.environ.get('LLAMA_RESUME_VERIFY_SAVE') == '1':
                self.save(model, optimizer, progress, scheduler)
                print(json.dumps({'event': 'RESUME_CHECKPOINT_ROUNDTRIP_PASS', 'rank': dist.get_rank(), 'step': progress['update_step']}), flush=True)
            return progress

    worker_globals['DistributedCheckpointRotation'] = Rotation
