import torch
import numpy as np

def get_prediction_sets(scores: torch.Tensor, threshold: float):
    """
    Args:
        scores: Tensor of shape (Nt, C) containing 1 - probabilities
        threshold: scalar threshold t

    Returns:
        List of length Nt, where each element is a list of class indices
        such that scores[i, j] <= threshold
    """
    Nt = scores.size(0)

    # Step 1: boolean mask
    mask = scores <= threshold

    # Step 2: get indices of True entries
    rows, cols = mask.nonzero(as_tuple=True)

    # Step 3: count how many per row
    counts = torch.bincount(rows, minlength=Nt)

    # Step 4: split column indices accordingly
    grouped = torch.split(cols, counts.tolist())

    # Step 5: convert to Python list of lists
    result = [g.tolist() for g in grouped]

    return result

def calculate_conformal_value(scores, alpha, mode="upper"):

    # Number of samples
    N = scores.shape[0]
 
    # Calculate sample index corresponding to target quantile
    k_exact = (N + 1) * (1 - alpha)
    if mode in ["lower", "rand"]:
        k       = int(np.floor(k_exact).item())
        resd    = k_exact - k
        # Correction for residual
        if mode == "rand":
            k   += int(np.random.choice([0,1], p=[1-resd, resd]).item())
    elif mode == "upper":
        k = int(np.ceil(k_exact).item())

    # Return quantile value
    if k/N > 1:
        return torch.inf
    else:
        return torch.kthvalue(scores, k, dim=0).values.to(scores.device)

