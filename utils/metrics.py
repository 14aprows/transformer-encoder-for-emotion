import torch 

def accuracy(pred, label):
    pred_class = torch.argmax(pred, dim=1)
    correct = (pred_class == label)
    return correct.float().mean().item()