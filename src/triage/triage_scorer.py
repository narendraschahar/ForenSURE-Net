import torch
from src.uncertainty.mc_dropout import mc_dropout_predict

def score_triage(model, images, device, temperature, mc_passes=10):
    \"\"\"
    Computes triage score for a batch of images.
    Triage Score = P_stego * Reliability * (1 - Uncertainty)
    \"\"\"
    mean_prob, uncertainty = mc_dropout_predict(model, images, device=device, mc_passes=mc_passes)

    # Reliability from temperature-scaled confidence
    logits = torch.logit(torch.tensor(mean_prob).clamp(1e-6, 1 - 1e-6)) / temperature
    reliability = torch.sigmoid(logits).numpy()

    triage_score = mean_prob * reliability * (1 - uncertainty)
    
    return mean_prob, reliability, uncertainty, triage_score
