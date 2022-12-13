# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp


class SimCLR(nn.Module):
    """
    Build a SimCLR model with a base encoder, and two MLPs
   
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=2048, T=1.0, cifar_head=False, 
    loss_type='dcl', N=50000, num_proj_layers=2, K=65536, m=0.999, radius=0.6, device=None):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        cifar_head: special input layers for training cifar datasets (default: false)
        loss_type: dynamatic contrastive loss (dcl) or contrastive loss (cl) (default: dcl)
        N: number of samples in the dataset used for computing moving average (default: 50000)
        num_proj_layers: number of non-linear projection head (default: 2)
        """
        super(SimCLR, self).__init__()
        self.T = T
        self.N = N
        self.loss_type = loss_type
        self.m = m
        self.K = K
        self.radius = 0.5
        
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.key_encoder = base_encoder(num_classes=mlp_dim)
        
        # build non-linear projection heads
        self._build_projector_and_predictor_mlps(dim, mlp_dim)
        
        # update input heads if image_size < 32 (e.g., cifar)
        print ('cifar head:', cifar_head)
        self.base_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.base_encoder.maxpool = nn.Identity()
        self.key_encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.key_encoder.maxpool = nn.Identity()

        for param_q, param_k in zip(self.base_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        
        # sogclr 
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device 
        
        if self.loss_type == 'dcl' or self.loss_type == 'moco_cl':
            self.u = torch.zeros(N).reshape(-1, 1) #.to(self.device) 
             
        self.LARGE_NUM = 1e9

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.base_encoder.parameters(), self.key_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        '''
        TODO: retrieval augumentation, how to select negative samples to our queue
        '''
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        # print(keys.shape)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # print(ptr)
        # print(batch_size)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # print(self.queue[:, ptr:ptr + batch_size].shape)
        # print(keys.T.shape)
        # print('-'*20)
        if self.queue[:, ptr:ptr + batch_size].shape[1] < keys.shape[0]:
            keys, rest_keys = keys[:self.queue[:, ptr:ptr + batch_size].shape[1]], keys[self.queue[:, ptr:ptr + batch_size].shape[1]:]
            batch_size = keys.shape[0]
        else:
            rest_keys = None

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

        if rest_keys is not None:
            batch_size = rest_keys.shape[0]
            self.queue[:, ptr:ptr + batch_size] = rest_keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer

            self.queue_ptr[0] = ptr


    def key_selection(self, query, key):
        selected_key = None
        avg_dist = 0.0
        for q in query:
            
            for k in key:
                dist = (q-k).pow(2).sum().sqrt()
                if dist <= 2 * self.radius:
                    if selected_key is None:
                        selected_key = k.unsqueeze(0)
                    elif k.cpu().numpy() not in selected_key.cpu().numpy():
                        selected_key = torch.cat((selected_key, k.unsqueeze(0)), 0)
                avg_dist += dist.item() / key.shape[0]
            # print('Average distance is ', avg_dist)
        avg_dist /= q.shape[0]
        self.radius = avg_dist / 2
        if selected_key is None:
            return key
        return selected_key


    def moco_contrast(self, im_1, im_2, index=None, gamma=0.99, local_rank=0, distributed=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q1 = self.base_encoder(im_1)  # queries: NxC
        q1 = nn.functional.normalize(q1, dim=1)
        q2 = self.base_encoder(im_2)  # queries: NxC
        q2 = nn.functional.normalize(q2, dim=1)
        batch_size = im_1.shape[0]

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k1 = self.key_encoder(im_2)  # keys: NxC
            k1 = nn.functional.normalize(k1, dim=1)
            k2 = self.key_encoder(im_1)  # keys: NxC
            k2 = nn.functional.normalize(k2, dim=1)

            # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_1 = torch.einsum('nc,nc->n', [q1, k1]).unsqueeze(-1) / self.T
        l_pos_2 = torch.einsum('nc,nc->n', [q2, k2]).unsqueeze(-1) / self.T
        # negative logits: NxK
        l_neg_1 = torch.einsum('nc,ck->nk', [q1, self.queue.clone().detach()]) / self.T
        l_neg_2 = torch.einsum('nc,ck->nk', [q2, self.queue.clone().detach()]) / self.T

        # logits: Nx(1+K)
        logits_1 = torch.cat([l_pos_1, l_neg_1], dim=1)
        logits_2 = torch.cat([l_pos_2, l_neg_2], dim=1)

        # labels: Nx(1+k) with all-one vectors ath 0th colum and rest of the entries are 0
        labels = F.one_hot(torch.zeros(batch_size, dtype=torch.long), 1+self.K).to(f'cuda:{local_rank}') 
        neg_masks = 1 - labels

        neg_logits_1 = torch.exp(logits_1/self.T) * neg_masks
        neg_logits_2 = torch.exp(logits_2/self.T) * neg_masks

        # u1 = (1 - gamma) * self.u[index].cuda(local_rank, non_blocking=True) + gamma * torch.sum(neg_logits_1, dim=1, keepdim=True)/(2*(batch_size-1))
        # u2 = (1 - gamma) * self.u[index].cuda(local_rank, non_blocking=True) + gamma * torch.sum(neg_logits_2, dim=1, keepdim=True)/(2*(batch_size-1))
        u1 = (1 - gamma) * self.u[index].cuda(local_rank, non_blocking=True) + gamma * torch.sum(neg_logits_1, dim=1, keepdim=True)/(self.K)
        u2 = (1 - gamma) * self.u[index].cuda(local_rank, non_blocking=True) + gamma * torch.sum(neg_logits_2, dim=1, keepdim=True)/(self.K)
        self.u[index] = u1.detach().cpu()+ u2.detach().cpu()

        p_neg_weights1 = (neg_logits_1/u1).detach()
        p_neg_weights2 = (neg_logits_2/u2).detach()

        def softmax_cross_entropy_with_logits(labels, logits, weights):
            expsum_neg_logits = torch.sum(weights*logits, dim=1, keepdim=True)/(self.K)
            normalized_logits = logits - expsum_neg_logits
            return -torch.sum(labels * normalized_logits, dim=1)
        
        loss1 = softmax_cross_entropy_with_logits(labels, logits_1, p_neg_weights1)
        loss2 = softmax_cross_entropy_with_logits(labels, logits_2, p_neg_weights2)
        

        # apply temperature
        # logits /= self.T

        # labels: positive key indicators
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        # key = torch.cat((self.key_selection(q1, k1), self.key_selection(q2, k2)), 0)
        # self._dequeue_and_enqueue(key)
        #self._dequeue_and_enqueue(self.key_selection(q1, k1))
        #self._dequeue_and_enqueue(self.key_selection(q2, k2))
        
        self._dequeue_and_enqueue(k1)
        self._dequeue_and_enqueue(k2)

        loss = (loss1 + loss2).mean()
        return loss
        # return logits, labels

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass
    
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def dynamic_contrastive_loss(self, hidden1, hidden2, index=None, gamma=0.99, distributed=True):
        # Get (normalized) hidden1 and hidden2.
        # hidden1.shape := [64, 3, 32, 32]
        hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(hidden2, p=2, dim=1)
        batch_size = hidden1.shape[0]
        
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).to(self.device) 
        masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).to(self.device) 

        logits_aa = torch.matmul(hidden1, hidden1_large.T)
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T)
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T)
        logits_ba = torch.matmul(hidden2, hidden1_large.T)

        #  SogCLR
        neg_mask = 1-labels
        logits_ab_aa = torch.cat([logits_ab, logits_aa ], 1) 
        logits_ba_bb = torch.cat([logits_ba, logits_bb ], 1) 
        # print(logits_ab_aa.shape)
      
        neg_logits1 = torch.exp(logits_ab_aa/self.T)*neg_mask   #(B, 2B)
        neg_logits2 = torch.exp(logits_ba_bb/self.T)*neg_mask
        
        # print(torch.sum(neg_logits1, dim=1, keepdim=True).shape)
        # print(self.u[index].shape)
        u1 = (1 - gamma) * self.u[index].cuda() + gamma * torch.sum(neg_logits1, dim=1, keepdim=True)/(2*(batch_size-1))
        u2 = (1 - gamma) * self.u[index].cuda() + gamma * torch.sum(neg_logits2, dim=1, keepdim=True)/(2*(batch_size-1))
        
        self.u[index] = u1.detach().cpu()+ u2.detach().cpu()

        p_neg_weights1 = (neg_logits1/u1).detach()
        p_neg_weights2 = (neg_logits2/u2).detach()

        def softmax_cross_entropy_with_logits(labels, logits, weights):
            expsum_neg_logits = torch.sum(weights*logits, dim=1, keepdim=True)/(2*(batch_size-1))
            normalized_logits = logits - expsum_neg_logits
            return -torch.sum(labels * normalized_logits, dim=1)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa, p_neg_weights1)
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb, p_neg_weights2)
        loss = (loss_a + loss_b).mean()

        return loss
    
    def contrastive_loss(self, hidden1, hidden2, index=None, gamma=None, distributed=True):
        # Get (normalized) hidden1 and hidden2.
        hidden1, hidden2 = F.normalize(hidden1, p=2, dim=1), F.normalize(hidden2, p=2, dim=1)
        batch_size = hidden1.shape[0]
       
        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size * 2).to(self.device) 
        masks  = F.one_hot(torch.arange(batch_size, dtype=torch.long), batch_size).to(self.device) 

        logits_aa = torch.matmul(hidden1, hidden1_large.T)/ self.T
        logits_aa = logits_aa - masks * self.LARGE_NUM
        logits_bb = torch.matmul(hidden2, hidden2_large.T)/ self.T
        logits_bb = logits_bb - masks * self.LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.T)/ self.T
        logits_ba = torch.matmul(hidden2, hidden1_large.T)/ self.T

        logits_ab_aa = torch.cat([logits_ab, logits_aa ], 1) 
        logits_ba_bb = torch.cat([logits_ba, logits_bb ], 1) 

        def softmax_cross_entropy_with_logits(labels, logits):
            #logits = logits - torch.max(logits)
            expsum_neg_logits = torch.sum(torch.exp(logits), dim=1, keepdim=True)
            normalized_logits = logits - torch.log(expsum_neg_logits)
            return -torch.sum(labels * normalized_logits, dim=1)

        loss_a = softmax_cross_entropy_with_logits(labels, logits_ab_aa)
        loss_b = softmax_cross_entropy_with_logits(labels, logits_ba_bb)
        loss = (loss_a + loss_b).mean()
        return loss
    
    def forward(self, x1, x2, index, gamma, local_rank=0):
        # compute features
        with amp.autocast():
            if self.loss_type == 'moco_cl':
                loss = self.moco_contrast(x1, x2, index, gamma, local_rank)
            else:
                h1 = self.base_encoder(x1)
                h2 = self.base_encoder(x2)

                if self.loss_type == 'dcl':
                    loss = self.dynamic_contrastive_loss(h1, h2, index, gamma) 
                elif self.loss_type == 'cl':   
                    loss = self.contrastive_loss(h1, h2)

        return loss

    # @torch.no_grad()
    # def _dequeue_and_enqueue(self, keys):
    #     # gather keys before updating queue
    #     keys = concat_all_gather(keys)

    #     batch_size = keys.shape[0]

    #     ptr = int(self.queue_ptr)
    #     # assert self.K % batch_size == 0  # for simplicity

    #     # replace the keys at ptr (dequeue and enqueue)
    #     self.queue[:, ptr:ptr + batch_size] = keys.T
    #     ptr = (ptr + batch_size) % self.K  # move pointer

    #     self.queue_ptr[0] = ptr


class SimCLR_ResNet(SimCLR):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim, num_proj_layers=2):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc  # remove original fc layer
            
        # projectors
        self.base_encoder.fc = self._build_mlp(num_proj_layers, hidden_dim, mlp_dim, dim)
        self.key_encoder.fc = self._build_mlp(num_proj_layers, hidden_dim, mlp_dim, dim)
   
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # print(torch.distributed.get_world_size())

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]

    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output