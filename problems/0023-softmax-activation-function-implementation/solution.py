import torch
import torch.nn.functional as F

def softmax(scores: list[float]) -> list[float]:
    """
    Compute the softmax activation function using PyTorch's built-in API.
    Input:
      - scores: list of floats (logits)
    Returns:
      - list of floats representing the softmax probabilities.
    """
    # Your implementation here
    tensor = torch.tensor(scores, dtype=torch.float32)
    x = torch.softmax(tensor, dim=0)
    x = x.tolist()
    return x
    pass
