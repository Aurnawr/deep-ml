import torch
import torch.nn.functional as F

def train_neuron(
    features: torch.Tensor,
    labels: torch.Tensor,
    initial_weights: torch.Tensor,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> tuple[list[float], float, list[float]]:

    weights = initial_weights.detach().clone().requires_grad_(True)
    bias = torch.tensor(
        initial_bias, dtype=weights.dtype, requires_grad=True
    )

    optim = torch.optim.SGD([weights, bias], lr=learning_rate)
    mse_values = []

    for _ in range(epochs):
        # Forward pass using current weights and bias
        predictions = torch.sigmoid(features @ weights + bias)

        # Compute MSE before the parameter update
        loss = F.mse_loss(predictions, labels)
        mse_values.append(round(loss.item(), 4))

        # Backpropagation and full-batch gradient descent
        optim.zero_grad()
        loss.backward()
        optim.step()

    updated_weights = [round(w, 4) for w in weights.detach().tolist()]
    updated_bias = round(bias.item(), 4)

    return updated_weights, updated_bias, mse_values