import torch
import torch.nn as nn
import torch.nn.functional as F


## iemocap loss function: same with CE loss
class MaskedCELoss(nn.Module):

    def __init__(self):
        super(MaskedCELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target, umask, mask_m=None, first_stage=True):
        """
        pred -> [batch*seq_lentrain_transformer_expert_missing_softmoe.py, n_classes]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        if first_stage:
            umask = umask.view(-1,1) # [batch*seq_len, 1]
            mask = umask.clone()

            if mask_m == None:
                mask_m = mask
            mask_m = mask_m.reshape(-1, 1)  # [batch*seq_len, 1]

            target = target.view(-1,1) # [batch*seq_len, 1]
            pred = F.log_softmax(pred, 1) # [batch*seqlen, n_classes]
            loss = self.loss(pred*mask*mask_m, (target*mask*mask_m).squeeze().long()) / torch.sum(mask*mask_m)
            return loss
        else:
            assert first_stage == False
            umask = umask.view(-1, 1)  # [batch*seq_len, 1]
            mask = umask.clone()

            # l = mask.size(0)//7
            # mask[:4*l] = 0
            # mask[1*l:] = 0

            if mask_m == None:
                mask_m = mask
            mask_m = mask_m.reshape(-1, 1)  # [batch*seq_len, 1]

            target = target.view(-1, 1)  # [batch*seq_len, 1]
            pred = F.log_softmax(pred, 1)  # [batch*seqlen, n_classes]
            loss = self.loss(pred * mask * mask_m, (target * mask * mask_m).squeeze().long()) / torch.sum(mask * mask_m)
            if torch.isnan(loss) == True:
                loss = 0
            return loss


## for cmumosi and cmumosei loss calculation
class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target, umask):
        """
        pred -> [batch*seq_len]
        target -> [batch*seq_len]
        umask -> [batch, seq_len]
        """
        umask = umask.view(-1, 1)  # [batch*seq_len, 1]
        mask = umask.clone()

        pred = pred.view(-1, 1) # [batch*seq_len, 1]
        target = target.view(-1, 1) # [batch*seq_len, 1]

        loss = self.loss(pred*mask, target*mask) / torch.sum(mask)

        return loss
