import subprocess
from pathlib import Path


def run(cmd):
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("experiments/checkpoints").mkdir(parents=True, exist_ok=True)
    Path("experiments/calibration").mkdir(parents=True, exist_ok=True)

    run(["python", "scripts/create_splits.py", "--seed", "42"])
    run(["python", "scripts/train_residual.py"])
    run(["python", "scripts/evaluate_residual.py"])
    run(["python", "scripts/calibrate_residual.py"])
    run(["python", "scripts/evaluate_uncertainty.py"])
    run(["python", "scripts/evaluate_case_folder.py"])
    run(["python", "scripts/evaluate_robustness.py"])
    run(["python", "scripts/generate_result_tables.py"])
    run(["python", "scripts/generate_figures.py"])
    run(["python", "scripts/generate_model_comparison.py"])

    print("\nFull pipeline completed successfully.")


if __name__ == "__main__":
    main()