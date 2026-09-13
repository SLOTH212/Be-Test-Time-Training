import re
QA_RE = re.compile(r"Question (\d+):\n(.*?)\n\nAnswer \1:\n(.*?)\n\n", re.S)

def render_qa(context,q):
    head="Context:\n\nPassage 1:\n"+context+"\n\n\nQuestions and Answers:\n\n"
    return head+"Question 1:\n"+q["question"]+"\n\nAnswer 1:\n"+q["answer"]+"\n\n"


def valid_qas(raw):
    ctx=raw.get("context","");out=[]
    for qa in raw.get("qas",[]):
        qid=str(qa.get("qid") or qa.get("id"));grounded=None
        for det in qa.get("detected_answers",[]):
            ans=det.get("text","")
            for span in det.get("char_spans",[]):
                s,e=int(span[0]),int(span[1])
                if 0<=s<=e<len(ctx) and ctx[s:e+1]==ans:grounded=ans;break
            if grounded is not None:break
        if grounded is not None:out.append({"qid":qid,"question":qa["question"],"answer":grounded})
    return out


def answer_spans(text):
    delim = "\n\nQuestions and Answers:\n\n"; base = text.rfind(delim)
    if base < 0: raise ValueError("QA template missing")
    tail = text[base + len(delim):]; spans = []
    for m in QA_RE.finditer(tail): spans.append([base + len(delim) + m.start(3), base + len(delim) + m.end(3)])
    if not spans: raise ValueError("answers missing")
    return spans


def token_answer_ranges(tok, text):
    spans = answer_spans(text); enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    mask = [any(a < e and b > s for s, e in spans) for a, b in enc["offset_mapping"]]
    ranges, start = [], None
    for i, v in enumerate(mask + [False]):
        if v and start is None: start = i
        elif not v and start is not None: ranges.append([start, i]); start = None
    return enc["input_ids"], ranges
