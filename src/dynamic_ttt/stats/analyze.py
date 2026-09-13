"""Descriptive tables from sanitized public CSVs; no benchmark text needed."""
import argparse,csv,json,statistics
from pathlib import Path
def summarize(path):
 rows=list(csv.DictReader(Path(path).open(encoding='utf-8')))
 if not rows:raise ValueError('EMPTY_RESULTS')
 numeric={}
 for field in rows[0]:
  try:values=[float(r[field]) for r in rows if r[field]!='']
  except ValueError:continue
  if values:numeric[field]={'n':len(values),'mean':statistics.fmean(values)}
 return {'rows':len(rows),'means':numeric}
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output');x=p.parse_args();text=json.dumps(summarize(x.input),indent=2,allow_nan=False)
 if x.output:Path(x.output).write_text(text+'\n',encoding='utf-8')
 else:print(text)
if __name__=='__main__':main()
