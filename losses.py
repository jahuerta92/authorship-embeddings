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

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, t_0=0.07, eps=1e-8):
        super(SupConLoss, self).__init__()
        self.temperature = torch.nn.Parameter(torch.tensor([t_0]))
        self.epsilon = eps


    def forward(self, features, labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
        Returns:
            A loss scalar.
        """
        batch_size = features.shape[0]

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(features.device)

        views = features.shape[1] # = n_views
        full_features = torch.cat(torch.unbind(features, dim=1), dim=0) # = [bsz*views, ...]

        # compute logits (cosine sim)
        anchor_dot_contrast = torch.matmul(F.normalize(full_features),
                                           F.normalize(full_features.T)) * torch.exp(self.temperature).clamp(100) # = [bsz*views, bsz*views]

        loss_0 = self._loss_from_dot(anchor_dot_contrast, mask, views, batch_size)
        loss_1 = self._loss_from_dot(anchor_dot_contrast.T, mask.T, views, batch_size)

        return (loss_0 + loss_1) / 2

    def _loss_from_dot(self, anchor_dot_contrast, mask, views, batch_size): #(anchor, contrast)
              # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(views, views)
        # mask-out self-contrast cases
        logits_mask = 1 - torch.eye(views*batch_size, device=mask.device)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.epsilon)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos.view(views, batch_size).mean()

        return loss

class InfoNCELoss(torch.nn.Module):
    def __init__(self, t_0=0.07, eps=1e-8):
        super(InfoNCELoss, self).__init__()
        self.temperature = torch.nn.Parameter(torch.tensor([t_0]))

    def forward(self, anchors, replicas):
        batch_size = anchors.shape[0]
        logits = (F.normalize(anchors) @ F.normalize(replicas.T)) * torch.exp(self.temperature).clamp(max=100)
        gt = torch.arange(0, batch_size, device=logits.device) 

        loss = (F.cross_entropy(logits.T, gt).mean() +
                F.cross_entropy(logits, gt).mean()) / 2

        with torch.no_grad():
            preds = logits.argmax(-1)
            preds_t = logits.T.argmax(-1)

            accuracy = (torch.sum(preds == gt) +
                        torch.sum(preds_t == gt)) / (batch_size * 2)

        return loss, accuracy
