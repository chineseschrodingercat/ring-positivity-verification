"""Replay all archived order-five decisions; run from this folder."""
from pathlib import Path
import json
from cp5 import membership

root=Path(__file__).resolve().parent
cases=json.loads((root/'membership_checks.json').read_text(encoding='utf8'))
for name,case in cases.items():
    got=membership(case['input'])
    assert json.loads(json.dumps(got))==case['result'],name
for name,case in json.loads((root/'gram_checks.json').read_text(encoding='utf8')).items():
    got=membership(case['A'])
    assert got['cp'] and got['interior_real_events_tested']==case['events'],name
print("All archived CP5 decisions reproduced.")
