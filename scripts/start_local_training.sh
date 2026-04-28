#!/bin/bash

# start_local_training.sh
# Initiates the full ForenSURE-Net SRNet training pipeline locally in the background.

LOGFILE="training_log.txt"

echo "========================================="
echo " Starting Local TIFS-Level SRNet Training "
echo "========================================="

echo "[1/3] Generating 0.4 bpp Stego Dataset..."
python scripts/generate_lsb_stego.py

echo "[2/3] Generating Train/Val/Test Splits..."
python scripts/create_splits.py --seed 42

echo "[3/3] Launching Background Training..."
echo "Training output is being redirected to $LOGFILE"
echo "You can monitor the progress by running: tail -f $LOGFILE"

# Run training in background with nohup
nohup python scripts/train_residual.py > $LOGFILE 2>&1 &

echo "========================================="
echo " Training Successfully Launched!"
echo " The SRNet is now optimizing in the background."
echo " PID: $!"
echo "========================================="
