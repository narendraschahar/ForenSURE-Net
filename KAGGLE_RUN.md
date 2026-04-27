# Kaggle Run Guide for ForenSURE-Net

This project is fully configured to be run on Kaggle using the Kaggle Notebook environment.

## Dataset Path

The BOSSbase 1.01 dataset should be added to your Kaggle environment. Ensure the dataset is mounted at the following path:
```text
/kaggle/input/datasets/narendrachahar/bossbase-1-01/BOSSbase_1.01
```

## Running the Pipeline on Kaggle

1. **Import the Notebook**: Upload the provided `notebooks/Kaggle_Pipeline.ipynb` file into Kaggle as a new Notebook.
2. **Add the Dataset**: Click on `Add Data` -> `Your Datasets` and add the `BOSSbase 1.01` dataset.
3. **Hardware Selection**: Choose the **GPU P100** or **T4x2** accelerator in Kaggle's Session options for faster training.
4. **Execute**: Run all cells in the notebook.

The notebook will automatically:
- Install required dependencies.
- Link the Kaggle dataset to the expected local directory `data/BOSSBase`.
- Run the full pipeline (`scripts/run_full_pipeline.py`) including data splitting, training, evaluation, and calibration.
- Output all results into the `results/` and `figures/` directories, which you can download from Kaggle's `/kaggle/working/` directory.