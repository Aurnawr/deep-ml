import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix using PyTorch.
    
    Args:
        a: A 2D matrix (can be list, numpy array, or torch.Tensor)
    
    Returns:
        A transposed torch.Tensor
    """
    a_t = torch.as_tensor(a)
    # Your code here
    b_t = torch.zeros(a_t.size(1),a_t.size(0))
    for i in range (a_t.size(1)):
        for k in range(a_t.size(0)):
            b_t[i][k]= a_t[k][i]

    return b_t
    pass