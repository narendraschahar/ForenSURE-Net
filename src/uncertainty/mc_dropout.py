import torch
import numpy as np


def enable_dropout(model):
    """
    Enable dropout layers during inference for MC Dropout.
    """
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


def mc_dropout_predict(model, images, device, mc_passes=10):
    """
    Returns:
    - mean probability
    - uncertainty score using probability variance
    """
    model.eval()
    enable_dropout(model)

    probs = []

    with torch.no_grad():
        for _ in range(mc_passes):
            logits = model(images.to(device)).squeeze(1)
            prob = torch.sigmoid(logits)
            probs.append(prob.cpu().numpy())

    probs = np.stack(probs, axis=0)

    mean_prob = np.mean(probs, axis=0)
    uncertainty = np.var(probs, axis=0)

    return mean_prob, uncertainty