import json
from pathlib import Path

Path("results").mkdir(parents=True, exist_ok=True)

ablation_plan = {
    "baseline_cnn": {
        "high_pass": False,
        "calibration": False,
        "uncertainty": False,
        "triage": False,
        "status": "ready"
    },
    "residual_stegnet": {
        "high_pass": True,
        "calibration": False,
        "uncertainty": False,
        "triage": False,
        "status": "ready"
    },
    "residual_plus_calibration": {
        "high_pass": True,
        "calibration": True,
        "uncertainty": False,
        "triage": False,
        "status": "ready"
    },
    "residual_plus_uncertainty": {
        "high_pass": True,
        "calibration": True,
        "uncertainty": True,
        "triage": False,
        "status": "ready"
    },
    "forensure_net_full": {
        "high_pass": True,
        "calibration": True,
        "uncertainty": True,
        "triage": True,
        "status": "ready"
    }
}

with open("results/ablation_plan.json", "w") as f:
    json.dump(ablation_plan, f, indent=4)

print("Saved: results/ablation_plan.json")
print(json.dumps(ablation_plan, indent=4))