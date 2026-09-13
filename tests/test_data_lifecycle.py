import pytest
from dynamic_ttt.data.lifecycle import require_audit

def test_empty_or_failed_audit_cannot_finalize(monkeypatch):
 import io
 from pathlib import Path
 for text in ['{"status":"PASS","data":{"records":0}}','{"status":"FAIL","data":{"records":10}}']:
  monkeypatch.setattr(Path,'open',lambda *a,**k:io.StringIO(text))
  with pytest.raises(ValueError):require_audit('audit.json')
