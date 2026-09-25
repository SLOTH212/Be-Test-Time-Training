"""Deterministic clean package, fresh extraction, manifest and executable tests."""
from pathlib import Path
import gzip,hashlib,json,os,shutil,subprocess,sys,tarfile,tempfile
W=Path(__file__).resolve().parents[1]
DEST=Path('/home/USER/ttt/packages/llama31_ntp_port_v1')
NAME='llama31_ntp_port_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
status=json.loads((W/'audits/PREPACKAGE_STATUS.json').read_text())
assert all(v=='PASS' for k,v in status.items() if k.endswith(('STATUS','PARITY','GUARD','LEVEL0','LEVEL1','LEVEL2','TEST','FLOW','ORACLE','ROUNDTRIP','SEMANTICS','COMPATIBILITY','FREEZE','AUTHORITY')))
# Exact all-source input checks immediately before release.
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
subprocess.run([sys.executable,str(W/'tests/test_level0.py')],env=env,check=True)
assert not DEST.exists(),'Refuse to overwrite an existing package'
roots=['reference','source','training','inference','adapters','configs','tests','audits','provenance','reports']
files=[p for root in roots for p in (W/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[W/'README.md']
classes=[]
for p in sorted(files):
    rel=str(p.relative_to(W))
    if rel.startswith('source/') and '/hf_llama/' in rel:category='LLAMA_MODEL_GLUE'
    elif rel.startswith(('source/','adapters/')):category='SHARED_GENERIC_ADAPTER'
    elif rel.startswith('tests/'):category='TEST'
    elif rel.startswith('configs/'):category='CONFIG'
    elif rel.endswith('.md'):category='DOCUMENTATION'
    else:category='PROVENANCE'
    classes.append(dict(path=rel,classification=category))
classes += [dict(path='provenance/FILE_CLASSIFICATION.json',classification='PROVENANCE'),dict(path='provenance/RELEASE_FILE_MANIFEST.json',classification='PROVENANCE')]
dump(W/'provenance/FILE_CLASSIFICATION.json',dict(files=classes,common_scientific_logic_change='Only positive configurable chunk-size validation in new ntp_core; all state/update arithmetic byte-identical. Frozen Qwen copies unchanged.'))
files.append(W/'provenance/FILE_CLASSIFICATION.json')
manifest=[dict(path=str(p.relative_to(W)),size=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
dump(W/'provenance/RELEASE_FILE_MANIFEST.json',dict(files=manifest,self_excluded=True,package_metadata_external=True))
files.append(W/'provenance/RELEASE_FILE_MANIFEST.json')
DEST.mkdir(parents=True)
archive=DEST/(NAME+'.tar.gz')
with archive.open('wb') as raw:
    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w',format=tarfile.PAX_FORMAT) as tar:
            for p in sorted(files):
                assert p.suffix not in ['.pt','.bin','.safetensors','.pyc']
                info=tar.gettarinfo(str(p),arcname=NAME+'/'+str(p.relative_to(W)))
                info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0;info.mode=0o644
                with p.open('rb') as f:tar.addfile(info,f)
expected_hash=sha(archive)
(DEST/(archive.name+'.sha256')).write_text(expected_hash+'  '+archive.name+'\n')
# New local verification directory; never transfers off this host.
verify=Path(tempfile.mkdtemp(prefix='llama31_ntp_port_verify_',dir=W/'package'))
with tarfile.open(archive,'r:gz') as tar:
    for member in tar.getmembers():
        assert member.isfile() and not member.issym() and '..' not in Path(member.name).parts and not Path(member.name).is_absolute()
    tar.extractall(verify,filter='data')
root=verify/NAME
for item in manifest:
    p=root/item['path'];assert p.stat().st_size==item['size'] and sha(p)==item['sha256'],item['path']
assert sha(root/'provenance/RELEASE_FILE_MANIFEST.json')==sha(W/'provenance/RELEASE_FILE_MANIFEST.json')
log=W/'package/extracted_tests.log'
with log.open('w') as stream:
    subprocess.run([sys.executable,str(root/'tests/run_tests.py'),'--extracted'],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
assert sha(archive)==expected_hash
extract_ports=json.loads((root/'audits/port_tests.json').read_text());assert extract_ports['status']=='PASS'
result=dict(status,LLAMA_NTP_PORT_STATUS='PASS_WITH_REAL_MODEL_SMOKE_DEFERRED',PACKAGE_PATH=str(archive),PACKAGE_SIZE=archive.stat().st_size,PACKAGE_SHA256=expected_hash,PACKAGE_EXTRACTION_STATUS='PASS',PACKAGE_HASH_PARITY='PASS',EXTRACTED_LLAMA_LEVEL0='PASS',EXTRACTED_LLAMA_LEVEL1='PASS',EXTRACTED_LLAMA_LEVEL2='PASS',EXTRACTION_ROOT=str(root),PACKAGE_FILES=len(files),LLAMA_NTP_PORT_INTEGRITY='PASS',SAFE_TO_BEGIN_LLAMA_HIT_GPU_VALIDATION='YES')
dump(W/'package/PACKAGE_VERIFICATION.json',result);dump(DEST/'PACKAGE_VERIFICATION.json',result)
print(json.dumps(result,indent=2))
