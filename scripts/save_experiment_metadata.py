import json
import shutil
from pathlib import Path
from datetime import datetime

CONFIG_PATH = Path("configs/lsb_residual_config.json")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path("experiments/runs") / f"{config['experiment_name']}_{run_id}"
run_dir.mkdir(parents=True, exist_ok=True)

shutil.copy2(CONFIG_PATH, run_dir / "config.json")

summary = {
    "run_id": run_id,
    "created_at": datetime.now().isoformat(),
    "config_file": str(CONFIG_PATH),
    "run_directory": str(run_dir),
    "note": "This stores experiment metadata for reproducibility."
}

with open(run_dir / "run_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("Experiment metadata saved to:")
print(run_dir)