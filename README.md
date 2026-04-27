# ForenSURE-Net

Reliability-Calibrated Steganalysis for Forensic Image Evidence Triage Under Domain Shift.

ForenSURE-Net is a state-of-the-art framework designed to detect steganography in images while providing calibrated confidence and uncertainty scores. This repository is specifically structured to run on Kaggle and includes a full pipeline for training, evaluation, and triage.

## Core Modules

The architecture is divided into the following primary modules:
- **Detection**: Core CNN models (e.g., `ResidualStegNet`, `BaselineCNN`) for image steganalysis.
- **Calibration**: Temperature scaling to ensure model probabilities correspond to real-world reliability.
- **Uncertainty**: MC Dropout techniques to evaluate the epistemic uncertainty of predictions.
- **Triage**: A scoring mechanism that combines P(stego), Reliability, and Uncertainty to rank suspicious images for human review.
- **Cross-domain Evaluation**: Robustness testing against transformations like JPEG compression, noise, and resizing.

## Project Structure

- `src/`: Core Python modules for the above architecture.
- `scripts/`: Entry-point scripts for each step of the pipeline.
- `notebooks/`: Contains `Kaggle_Pipeline.ipynb` for easy Kaggle execution.
- `configs/`: Experiment configuration files.

## Installation & Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare the data: Ensure the BOSSbase dataset cover and stego images are placed in `data/BOSSBase/cover/` and `data/BOSSBase/stego/`.

## Running the Full Pipeline

To execute the entire end-to-end process (data splitting, training, calibration, robustness evaluation, and figure generation), run:
```bash
python scripts/run_full_pipeline.py
```

## Forensic Tool Usage

You can use the built-in CLI to scan an arbitrary folder of images and generate a premium HTML Triage Report. This is designed for investigators running the tool locally.

```bash
# Example usage
python forensure_cli.py scan --dir /path/to/suspect/drive --weights experiments/checkpoints/residual_stegnet_best.pth --out forensic_report.html
```

## Kaggle Execution

Please see `KAGGLE_RUN.md` and use the `notebooks/Kaggle_Pipeline.ipynb` for instructions on how to easily run this project in a Kaggle environment with GPU acceleration.
