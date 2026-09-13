from dynamic_ttt.mechanisms.runtime import kp_plan
def test_factorial_state_sources(layers):
 p=kp_plan(['L0']+['L6']*15,2,layers)
 assert p['control_sequence']==['OFF']+['L6']*15
 assert p['cells']['F00']['W_source']=='NATIVE'
 assert p['cells']['F00']['K_source']=='CONTROL'
 assert p['cells']['F11']['K_source']=='NATIVE'
 assert p['cells']['F11']['injection_layers']==[0,6]
