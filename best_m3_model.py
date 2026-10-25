import math

import torch
from torch import nn
from torch.nn import functional as F


class SignalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(1, 16, 1)
        self.branch = nn.ModuleList(nn.Conv2d(4, 4, 1) for _ in range(3))
        self.merge = nn.ModuleList([nn.Conv2d(4, 4, 1), nn.Conv2d(6, 4, 1), nn.Conv2d(6, 4, 1)])
        self.eca = nn.Conv1d(1, 1, 3, padding=1, bias=False)
        self.project = nn.Sequential(nn.Conv1d(72, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU())
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, 256, .1, batch_first=True),
            1, enable_nested_tensor=False)

    def features(self, x):
        streams = self.expand(x.unsqueeze(1)).chunk(4, dim=1)
        outputs, carry = [streams[0]], None
        for i in range(3):
            branch = self.branch[i](streams[i+1])
            if carry is not None:
                branch = torch.cat([carry, branch], 1)
            branch = self.merge[i](branch)
            if i < 2:
                carry, emit = branch.chunk(2, 1)
                outputs.append(emit)
            else:
                outputs.append(branch)
        y = torch.cat(outputs, 1)
        weights = self.eca(y.mean((2, 3)).unsqueeze(1)).sigmoid().squeeze(1)
        y = y*weights[:, :, None, None]
        y = y.permute(0, 1, 3, 2).reshape(len(x), 72, x.shape[1])
        return F.avg_pool1d(self.project(y), 10, 10).transpose(1, 2)

    def encode(self, tokens):
        t = tokens.shape[1]
        position = torch.arange(t, device=tokens.device, dtype=tokens.dtype)[:, None]
        scales = torch.exp(torch.arange(0, 128, 2, device=tokens.device)*(-math.log(10000)/128))
        positional = torch.zeros(t, 128, device=tokens.device)
        positional[:, 0::2], positional[:, 1::2] = torch.sin(position*scales), torch.cos(position*scales)
        return self.transformer(tokens+positional)

    def forward(self, x):
        return self.encode(self.features(x))


class LocalGlobalFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(128, 384, bias=False)
        self.gate = nn.Linear(256, 1)

    def forward(self, x, labels):
        membership = F.one_hot(labels, num_classes=4).to(x.dtype)
        count = membership.sum(1)
        q, k, v = self.qkv(x).chunk(3, -1)
        scores = q @ k.transpose(1, 2)/(x.shape[-1]**.5)
        same = labels[:, :, None] == labels[:, None, :]
        intra = scores.masked_fill(~same, -float("inf")).softmax(-1) @ v
        centers = membership.transpose(1, 2) @ x/count[:, :, None].clamp_min(1)
        cq, ck, cv = self.qkv(centers).chunk(3, -1)
        scores = cq @ ck.transpose(1, 2)/(x.shape[-1]**.5)
        inter = scores.masked_fill(count[:, None, :] == 0, -float("inf")).softmax(-1) @ cv
        global_value = inter.gather(1, labels[..., None].expand(-1, -1, x.shape[-1]))
        gate = self.gate(torch.cat([intra, global_value], -1)).sigmoid()
        return gate*intra+(1-gate)*global_value


class M3Query(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(128, 4, 256, .1, batch_first=True),
            2, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(128, 4, 256, .1, batch_first=True), 2)
        self.queries = nn.Parameter(torch.randn(6, 128)*.1)
        self.actor_proj = nn.Linear(128, 128)
        self.group_proj = nn.Linear(128, 128)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 12))
        self.person_activity = nn.Linear(128, 11)
        self.person_action = nn.Linear(128, 10)
        self.local_norm = nn.LayerNorm(128)
        self.local_projection = nn.Linear(128, 128)
        self.frozen_local_norm = nn.LayerNorm(128)
        self.lgfe = LocalGlobalFeatures()
        self.global_gate = nn.Parameter(torch.zeros(128))
        self.direct_pair = nn.Sequential(nn.Linear(256, 32), nn.GELU(), nn.Linear(32, 1))
        self.direct_activity = nn.Linear(128, 11)

    def forward(self, x):
        if x.ndim != 3 or x.shape[-1] != 257:
            raise ValueError("Expected [batch, people, 257] signal components")
        local = x[..., :128]
        base = self.frozen_local_norm(local)
        labels = x[..., 256].round().long().clamp(0, 3)
        direct = self.frozen_local_norm(local+self.lgfe(x[..., 128:256], labels))
        base = base+self.global_gate*(direct-base)
        nodes = self.context(base+self.local_projection(self.local_norm(local)))
        query = self.decoder(self.queries[None].expand(len(x), -1, -1), nodes)
        membership = self.actor_proj(nodes)@self.group_proj(query).transpose(1, 2)/(128**.5)
        assignment = membership.softmax(-1)
        pooled = assignment.transpose(1, 2)@nodes/assignment.sum(1).unsqueeze(-1).clamp_min(1e-6)
        classes = self.classifier(torch.cat([query, pooled], -1))
        probability = .75*self.person_activity(nodes).softmax(-1)+.25*self.direct_activity(direct).softmax(-1)
        left, right = direct[:, :, None], direct[:, None, :]
        pair = self.direct_pair(torch.cat([(left-right).abs(), left*right], -1)).squeeze(-1)
        return (membership, classes, probability.clamp_min(1e-7).log(),
                self.person_action(nodes), pair, torch.cat([base, local], -1))


class RelationConfidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.nodes = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 48), nn.GELU())
        self.pair = nn.Sequential(nn.Linear(99, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, features, affinity, activity):
        z = self.nodes(features)
        left, right = z[:, :, None], z[:, None, :]
        agreement = activity @ activity.transpose(1, 2)
        entropy = -(activity.clamp_min(1e-7).log()*activity).sum(-1)
        entropy_pair = (entropy[:, :, None]+entropy[:, None, :])/2
        pair = torch.cat([(left-right).abs(), left*right, affinity[..., None],
                          agreement[..., None], entropy_pair[..., None]], -1)
        correction = 2*torch.tanh(self.pair(pair).squeeze(-1)/2)
        logits = torch.logit(affinity.clamp(1e-5, 1-1e-5))+correction
        return (logits+logits.transpose(1, 2))/2


class GroupClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(771, 128), nn.GELU(), nn.Dropout(.25),
                                  nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 11))

    def forward(self, x):
        return self.body(x)
