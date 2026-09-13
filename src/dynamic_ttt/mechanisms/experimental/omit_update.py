"""Engineering-only omission of one selected update; downstream replay required."""
from dynamic_ttt.eval.runtime import action_layers as layer_mapping
LAYERS=[0,6,12,18,24,30]
def action_layers(action): return layer_mapping(LAYERS)[action]

def deletion_target(seq,chunk,layer):
 if type(chunk) is not int or not 0<=chunk<len(seq) or layer not in action_layers(seq[chunk]):raise ValueError('DELETION_TARGET')
 return {'chunk_index_0':chunk,'layer':layer,'omit_update_only':True,'action_sequence':list(seq)}
