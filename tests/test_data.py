from dynamic_ttt.data.qa import render_qa,answer_spans,valid_qas
def test_grounding_and_template():
 raw={'context':'a blue cup','qas':[{'qid':'q','question':'color?','detected_answers':[{'text':'blue','char_spans':[[2,5]]}]}]}
 q=valid_qas(raw)[0];text=render_qa(raw['context'],q);s,e=answer_spans(text)[0];assert text[s:e]=='blue'
