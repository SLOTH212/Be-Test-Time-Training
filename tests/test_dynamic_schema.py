import json
from pathlib import Path
import jsonschema,pytest
from dynamic_ttt.dynamic.trajectory import validate
def test_schema():
 root=Path(__file__).resolve().parents[1];row=json.loads((root/'examples/trajectory.json').read_text())
 jsonschema.validate(row,json.loads((root/'schemas/dynamic_trajectory.schema.json').read_text()));validate(row)
 row['num_complete_chunks']=7
 with pytest.raises(ValueError):validate(row)
