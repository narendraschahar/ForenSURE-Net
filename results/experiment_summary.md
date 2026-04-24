# ForenSURE-Net Experiment Summary

## Project Title

ForenSURE-Net: Reliability-Calibrated Steganalysis for Forensic Image Evidence Triage Under Domain Shift

## Current Implementation Status

The current implementation includes:

1. Baseline CNN
2. ResidualStegNet with fixed high-pass filtering
3. Temperature scaling calibration
4. MC Dropout uncertainty estimation
5. Evidence triage score
6. Case-folder simulation
7. Robustness testing
8. Result tables
9. Publication figures

## Dataset Used

Current dataset:

- BOSSBase 1.01 cover images
- LSB-generated stego images

Important note:

The current stego images are generated using LSB embedding only. Therefore, current results are suitable for pipeline verification, but not sufficient for a high-quality journal submission.

## Current Model

ResidualStegNet:

- Fixed high-pass preprocessing layer
- CNN feature extractor
- Dropout-enabled classifier
- Binary stego/cover prediction

## Calibration

Temperature scaling is applied using validation logits.

Metrics:

- Expected Calibration Error
- Brier Score

## Uncertainty

MC Dropout is used during inference.

Output:

- Mean stego probability
- Predictive uncertainty

## Triage Formula

Triage Score = P_stego × Reliability × (1 - Uncertainty)

## Case-Folder Evaluation

The system simulates forensic folders containing mixed cover and stego images.

Metrics:

- Top-5 hit rate
- Top-10 hit rate
- Mean rank
- Median rank
- Best rank

## Robustness Evaluation

Transformations tested:

- JPEG quality 75
- JPEG quality 50
- Resize down-up
- Center crop
- Gaussian noise
- Screenshot-like degradation

## Current Limitation

The present results should not be reported as final research findings because:

1. LSB steganography is too simple.
2. Only one dataset source has been used.
3. No cross-domain evaluation has been completed yet.
4. No comparison with standard steganalysis methods has been added yet.
5. No statistical significance testing has been performed yet.

## Next Required Research Steps

1. Add standard steganographic algorithms:
   - WOW
   - S-UNIWARD
   - HUGO or MiPOD if feasible

2. Add cross-domain datasets:
   - BOWS2
   - ALASKA2
   - StegoAppDB

3. Add ablation studies:
   - Without calibration
   - Without uncertainty
   - Without triage
   - Without high-pass residual layer

4. Add comparison baselines:
   - Baseline CNN
   - ResidualStegNet
   - ForenSURE-Net full pipeline

5. Add statistical testing:
   - Mean ± standard deviation
   - Multiple random seeds
   - Confidence intervals