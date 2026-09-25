from pathlib import Path
import json, math, hashlib
R = Path(__file__).resolve().parents[1]
def load(name):
    return json.loads((R / "results" / name / "AGGREGATE.json").read_text())
def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-10), (a, b)
b = load("base"); f = load("fixed8_v2"); d = load("dynamic_v2_no_cwe")
assert len(b["per_sample"]) == len(f["per_sample"]) == 6500
assert len(d["per_sample"]) == d["samples"] == 6000
for data in [b, f, d]:
    ids = [x["sample_id"] for x in data["per_sample"]]
    assert len(ids) == len(set(ids))
close(sum(x["score"] for x in b["per_sample"]) / 6500, b["mean_score"])
for a, mean in f["action_means"].items():
    close(sum(x[a] for x in f["per_sample"]) / 6500, mean)
close(sum(x["sample_best"] for x in f["per_sample"]) / 6500, f["sample_best_mean"])
lookup = {x["sample_id"]: x for x in f["per_sample"]}
for x in d["per_sample"]:
    assert "cwe" not in x["task"]
    close(x["off"], lookup[x["sample_id"]]["OFF"])
    close(x["sample_best"], lookup[x["sample_id"]]["sample_best"])
for a, mean in d["means"].items():
    close(sum(x[a] for x in d["per_sample"]) / 6000, mean)
c = [x for x in d["per_sample"] if x["searched"]]
assert len(c) == d["searched"] == 570
up = sum(x["dynamic"] > x["sample_best"] + 1e-12 for x in c)
down = sum(x["dynamic"] < x["sample_best"] - 1e-12 for x in c)
assert (up, down) == (39, 0)
t = json.loads((R / "results/dynamic_v2_no_cwe/trajectories.json").read_text())
assert {x["sample_id"] for x in t} == {x["sample_id"] for x in c}
cl = {x["sample_id"]: x for x in c}
for x in t:
    close(x["dynamic_search_score"], cl[x["sample_id"]]["dynamic"])
if (R / "SHA256SUMS.txt").exists():
    for line in (R / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert hashlib.sha256((R / name).read_bytes()).hexdigest() == expected, name
print(json.dumps({"status": "PASS", "base_n": 6500, "dynamic_n": 6000,
                  "searched": len(c), "improved": up, "ties": len(c)-up-down,
                  "worse": down, "dynamic_means": d["means"]}, indent=2))
