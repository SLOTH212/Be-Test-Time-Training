def validate(row):
 required={'sample_id','task','num_complete_chunks','actions','action_space','score','model_identity','config_identity'}
 if set(row)!=required:raise ValueError('TRAJECTORY_FIELDS')
 if type(row['num_complete_chunks']) is not int or row['num_complete_chunks']<0 or len(row['actions'])!=row['num_complete_chunks']:raise ValueError('CHUNK_COUNT')
 if len(set(row['action_space']))!=len(row['action_space']) or any(a not in row['action_space'] for a in row['actions']):raise ValueError('ACTION_SPACE')
 import math
 if not math.isfinite(row['score']) or not 0<=row['score']<=1:raise ValueError('SCORE')
 return row
