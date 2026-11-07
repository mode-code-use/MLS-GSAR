"""Batched equivalent of joint_loss for contiguous synthetic group IDs."""
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F


def validate_group_labels(groups, activities, slots=6):
    for row, labels in zip(np.asarray(groups), np.asarray(activities)):
        ids = np.unique(row)
        if not np.array_equal(ids, np.arange(len(ids))) or len(ids) > slots:
            raise ValueError("Expected contiguous zero-based group IDs within query capacity")
        if any(len(np.unique(labels[row == group])) != 1 for group in ids):
            raise ValueError("Inconsistent group activity targets")


def batched_joint_loss(outputs, groups, activities, actions):
    membership, classes, person, action = outputs
    b, n, k = membership.shape
    masks = F.one_hot(groups, num_classes=k).to(membership.dtype)
    counts = masks.sum(1)
    group_labels = activities.gather(1, masks.argmax(1)).masked_fill(counts == 0, 11)
    group_labels = group_labels.masked_fill(group_labels < 0, 10)
    log_member, log_class = membership.log_softmax(-1), classes.log_softmax(-1)
    cost = -(log_member.transpose(1, 2) @ masks) / counts[:, None].clamp_min(1)
    cost = cost - log_class.gather(2, group_labels[:, None].expand(-1, k, -1))
    # One host transfer per batch; SciPy remains the matching engine.
    cpu_cost, cpu_counts, cpu_labels = cost.detach().cpu().numpy(), counts.detach().cpu().numpy(), group_labels.cpu().numpy()
    targets = np.full((b, k), 11, dtype=np.int64)
    slot_for_group = np.zeros((b, k), dtype=np.int64)
    for i in range(b):
        slots, columns = linear_sum_assignment(cpu_cost[i, :, :int((cpu_counts[i] > 0).sum())])
        targets[i, slots] = cpu_labels[i, columns]
        slot_for_group[i, columns] = slots
    class_targets = torch.as_tensor(targets, device=groups.device)
    member_targets = torch.as_tensor(slot_for_group, device=groups.device).gather(1, groups)
    weights = classes.new_ones(12)
    weights[11] = .2
    class_loss = F.cross_entropy(classes.flatten(0, 1), class_targets.flatten(), weight=weights)
    member_loss = F.cross_entropy(membership.flatten(0, 1), member_targets.flatten())
    labels = activities.masked_fill(activities < 0, 10)
    individual = F.cross_entropy(person.flatten(0, 1), labels.flatten())
    action_loss = F.cross_entropy(action.flatten(0, 1), actions.flatten())
    assignment = membership.softmax(-1)
    joint = assignment @ classes.softmax(-1)[:, :, :11]
    semantic = F.nll_loss(joint.clamp_min(1e-7).log().flatten(0, 1), labels.flatten())
    affinity = assignment @ assignment.transpose(1, 2)
    upper = torch.triu(torch.ones(n, n, dtype=torch.bool, device=groups.device), 1)
    truth = (groups[:, :, None] == groups[:, None, :])[:, upper]
    edge = F.binary_cross_entropy(affinity[:, upper].clamp(1e-6, 1-1e-6), truth.float(), reduction="none")
    terms = [edge[mask].mean() for mask in (truth, ~truth) if mask.any()]
    balanced = torch.stack(terms).mean()
    return class_loss+member_loss+.3*individual+.1*action_loss+.5*semantic+.5*balanced
