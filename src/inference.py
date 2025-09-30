import torch

def infer(model, dataloader, thres = 0.5, device='cpu'):
    """
    Perform inference on the given dataloader using the provided model.
    Args:
        model (torch.nn.Module): Trained model for inference.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset to infer on.
        device (str): Device to perform inference on ('cpu' or 'cuda').
    Returns:
        all_predictions (torch.Tensor): Predicted binary labels.
        all_probs (torch.Tensor): Predicted probabilities.
        all_true_labels (torch.Tensor): True binary labels.
    """
    model.eval()
    all_predictions = []
    all_probs = []
    all_true_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            probs = torch.sigmoid(outputs)
            preds = (probs > thres).int()

            all_predictions.append(preds)
            all_probs.append(probs)
            all_true_labels.append(labels)
            
    all_probs = torch.cat(all_probs, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_true_labels = torch.cat(all_true_labels, dim=0)

    return all_predictions, all_probs, all_true_labels


def get_pred_and_prob(model, input, device='cpu'):
    """
    Get prediction and probability for a single input using the provided model.
    Args:
        model (torch.nn.Module): Trained model for inference.
        input (torch.Tensor): Input tensor for which to get prediction and probability.
        device (str): Device to perform inference on ('cpu' or 'cuda').
    Returns:
        pred (torch.Tensor): Predicted binary label.
        prob (torch.Tensor): Predicted probability.
    """
    model.eval()
    with torch.no_grad():
        output = model(input.to(device))
        prob = torch.sigmoid(output)
        pred = (prob > 0.5).int()
        
    return pred, prob