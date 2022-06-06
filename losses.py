import torch
import torch.nn.functional as F

def oneway_infonce_loss(a, b, t, smoothing=0.0, labels=None):
    logits = (F.normalize(a) @ F.normalize(b.T)) * torch.exp(t).clamp(max=100)
    loss = F.cross_entropy(logits, labels, label_smoothing=smoothing).mean()

    with torch.no_grad():
            preds = logits.argmax(-1)
            accuracy = torch.sum(preds == labels) / len(a)
    
    return loss, accuracy

def infonce_loss(a, b, t, smoothing=0.0, labels=None):
    batch_size = a.shape[0]
    logits = (F.normalize(a) @ F.normalize(b.T)) * torch.exp(t).clamp(max=100)
    gt = torch.arange(0, batch_size, device=logits.device) 
    '''
    if labels is not None:
        loss_a = F.cross_entropy(logits.T, gt, label_smoothing=smoothing, reduction='none')[labels.long()].mean()
        loss_b = F.cross_entropy(logits, gt, label_smoothing=smoothing, reduction='none')[labels.long()].mean()

        loss = (loss_a + loss_b) / 2
    else:'''
    loss = (F.cross_entropy(logits.T, gt, label_smoothing=smoothing).mean() +
            F.cross_entropy(logits, gt, label_smoothing=smoothing).mean()) / 2

    with torch.no_grad():
        preds = logits.argmax(-1)
        preds_t = logits.T.argmax(-1)

        accuracy = (torch.sum(preds == gt) +
                    torch.sum(preds_t == gt)) / (batch_size * 2)

    return loss, accuracy

def flatnce_loss(a, b, t, smoothing=0.0, labels=None):
    #from https://github.com/Junya-Chen/FlatCLR/blob/main/flatclr.py

    batch_size = a.shape[0]
    logits = (F.normalize(a) @ F.normalize(b.T))# * torch.exp(t).clamp(max=100)
    labels = torch.arange(0, batch_size, device=logits.device)

    # discard the main diagonal from both: labels and similarities matrix
    mask = 1 - torch.eye(batch_size).to(logits.device) # Positive and negative example similarities
    logits_pos = torch.diagonal(logits).view(batch_size, -1) # Get positive similarities

    clogits_a = mask * (logits - logits_pos) * torch.exp(t).clamp(max=100, min=-100)
    clogits_b = mask * (logits.T - logits_pos) * torch.exp(t).clamp(max=100, min=-100) 

    sum_a = torch.logsumexp(clogits_a, dim=1) - 1 # To offset exp(0) per row
    sum_b = torch.logsumexp(clogits_b, dim=1) - 1 
    sum_clogits = torch.cat([sum_a, sum_b], dim=0)

    loss_vector = torch.exp(sum_clogits-sum_clogits.detach())
    
    with torch.no_grad():
        dummy_logits = logits * mask
        dummy_loss = (F.cross_entropy(dummy_logits.T, labels, label_smoothing=smoothing).mean() +
                      F.cross_entropy(dummy_logits, labels, label_smoothing=smoothing).mean()) / 2

    loss = loss_vector.mean() - 1 + dummy_loss

    with torch.no_grad():
        preds = logits.argmax(-1)
        preds_t = logits.T.argmax(-1)

        accuracy = (torch.sum(preds == labels) +
                    torch.sum(preds_t == labels)) / (batch_size * 2)

    return loss, accuracy
