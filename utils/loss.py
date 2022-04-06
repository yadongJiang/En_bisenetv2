from socketserver import DatagramRequestHandler
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ImageBasedCrossEntropyLoss2d(nn.Module):
    
    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), 
                    range(self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1 ## 每一类的权重，数量越多，则权重越大
        return hist

    def forward(self, inputs, targets):
        targets = targets.long()
        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]): # bs
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()
            
            loss += self.nll_loss(F.log_softmax(inputs[i].unsqueeze(0)),
                                          targets[i].unsqueeze(0))
        return loss

class JointEdgeSegLoss(nn.Module):
    def __init__(self, classes, weight=None, reduction='mean', ignore_index=255, 
                 norm=False, upper_bound=1.0, mode='train', 
                 edge_weight=1, seg_weight=1, att_weight=1, dual_weight=1, edge='none'):
        super(JointEdgeSegLoss, self).__init__()
        self.num_classes = classes
        self.seg_loss = ImageBasedCrossEntropyLoss2d(
                classes=classes, ignore_index=ignore_index, upper_bound=upper_bound).cuda()

    def edge_attention(self, inputs, targets, edgemask):
        n, c, h, w = inputs.size()
        filter = torch.ones_like(targets) * 255
        return self.seg_loss(inputs, 
                             torch.where(edgemask.max(1)[0] > 0, targets, filter))

    def forward(self, inputs, targets):
        segmask, edgemask = targets

        losses = {}
        losses['seg_loss'] = self.seg_loss(inputs, segmask)
        losses['att_loss'] = self.edge_attention(inputs, segmask, edgemask)
        loss_ = 0.0
        loss_ += losses['seg_loss']
        loss_ += losses['att_loss']

        return loss_