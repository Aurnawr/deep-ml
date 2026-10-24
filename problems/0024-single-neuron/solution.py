import torch
import torch.nn.functional as F

def single_neuron_model(features, labels, weights, bias):
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)

    predictions = torch.sigmoid(torch.matmul(features, weights) + bias)

    loss = F.mse_loss(predictions, labels)

    predictions = [round(p, 4) for p in predictions.tolist()]
    loss = round(loss.item(), 4)

    return predictions, loss