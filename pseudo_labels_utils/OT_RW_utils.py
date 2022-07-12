"""
We refer to FLOT (https://github.com/valeoai/FLOT/blob/master/flot/tools/ot.py)
when implementing the sinkhorn algorithm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from pointnet2 import pointnet2_utils as pointutils

def OT_RW(pc1, pc2, pred_flows, norm1, norm2, color1, color2,
          pc1_full, OT_iter=40, RW_step=5, is_infinite_step = False, Alpha = 0.8):
    """
    Pseudo label generation and refinement

    ----------
    Input:
        pc1, pc2: Input points position, [B, 3, N]
        pred_flows: Scene flow prediction, [B, 3, N]
        norm1, norm2: Surface normal, [B, 3, N]
        color1, color2: When color is available, the size is [B, 3, N]; otherwise, it's set to None.

        pc1_full: When pc1 and pc2 are sampled data, pc1_full is the original first point cloud, [B, 3, N_full];
                  otherwise, pc1_full is set to None.

        OT_iter: Number of unrolled iteration of the Sinkhorn algorithm, int

        RW_step: Number of random walk steps, int
        is_infinite_step: Whether to apply an infinite number of random walk steps, bool
        Alpha: Parameter to control the tradeoff between refinements and initial values in RW, int

    -------
    Returns:
        refined_pseudo_label: Refined dense pseudo labels
    """

    # Pre-warped first point cloud
    warped_pc1 = pred_flows + pc1

    # Pseudo Label Generation Module
    indices, valid_vector = OT_module(warped_pc1, pc2, norm1, norm2, color1, color2, OT_iter=OT_iter)

    # pseudo labels
    nn_pc2 = pointutils.grouping_operation(pc2, torch.unsqueeze(indices, 2).int().contiguous())
    nn_pc2 = torch.squeeze(nn_pc2, -1)
    pseudo_label = nn_pc2 - pc1

    # Pseudo Label Refinement Module
    refined_pseudo_label = RW_module(pc1, pseudo_label, RW_step, valid_vector, pc1_full, is_infinite_step, Alpha=Alpha)

    return refined_pseudo_label.permute(0, 2, 1)


def Cost_cosine_distance(norm1, norm2):
    # input [B, 3, N], [B, 3, N]
    norm_matrix = torch.matmul(norm1.transpose(2, 1), norm2)
    norm_matrix = torch.abs(norm_matrix)
    norm_cost_matrix = (1 - norm_matrix)
    return norm_cost_matrix


def Cost_Gaussian_function(data1, data2, theta_2=3, threshold_2=12):
    # input [B, 3, N], [B, 3, N]
    distance2_matrix = torch.sum(data1 ** 2, 1, keepdim=True).transpose(2, 1) # B, N, 1
    distance2_matrix = distance2_matrix + torch.sum(data2 ** 2, 1, keepdim=True)  # B, N, N
    distance2_matrix = distance2_matrix - 2 * torch.matmul(data1.transpose(2, 1), data2)

    support = distance2_matrix < threshold_2
    distance_cost_matrix = 1 - torch.exp(-distance2_matrix / theta_2)
    return distance_cost_matrix, support


def OT(C, epsilon=0.03, OT_iter=4):
    B, N1, N2 = C.shape

    # Entropic regularisation
    K = torch.exp( -C / epsilon)

    # Init. of Sinkhorn algorithm
    a = torch.ones((B, N1, 1), device=C.device, dtype=C.dtype) / N1
    prob1 = torch.ones((B, N1, 1), device=C.device, dtype=C.dtype) / N1
    prob2 = torch.ones((B, N2, 1), device=C.device, dtype=C.dtype) / N2

    # Sinkhorn algorithm
    for _ in range(OT_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = prob2 / (KTa + 1e-8)

        # Update a
        Kb = torch.bmm(K, b)
        a = prob1 / (Kb + 1e-8)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))
    return T


def OT_module(pc1, pc2, norm1, norm2, color1, color2, OT_iter=4):
    """
    Pseudo label generation

    ----------
    Returns:
        indices: index of corresponding point, [B, N]
        valid_vector: binary vector with zeros indicating
                      where its corresponding point is invalid
                      and this point will be treated as unlabeled, [B, N]
    """

    # Matching cost
    Cost_dist, Support = Cost_Gaussian_function(pc1, pc2)
    Cost_norm = Cost_cosine_distance(norm1, norm2)
    Cost = Cost_dist + Cost_norm

    if color1 is not None:
        Cost_color, _ = Cost_Gaussian_function(color1, color2, theta_2=0.12)
        Cost = Cost + Cost_color

    # Optimal transport plan
    T = OT(Cost, epsilon=0.03, OT_iter=OT_iter)

    # Hard correspondence matrix
    indices = T.max(2).indices
    matrix2 = torch.nn.functional.one_hot(indices, num_classes=T.shape[2])

    # Remove some invalid correspondences with large displacements
    valid_map = matrix2 * Support
    valid_vector = torch.sum(valid_map, 2)

    return indices, valid_vector.float()


def RW_weight(pc1, theta_2 = 0.8):
    distance2_matrix = torch.sum(pc1 ** 2, 1, keepdim=True).transpose(2, 1) # B, N, 1
    distance2_matrix = distance2_matrix + torch.sum(pc1 ** 2, 1, keepdim=True)  # B, N, N
    distance2_matrix = distance2_matrix - 2 * torch.matmul(pc1.transpose(2, 1), pc1)

    weight_matrix = torch.exp(-distance2_matrix / theta_2)

    # Set diagonal elements to zero
    invalid_mask = torch.eye(pc1.shape[-1]).cuda()
    valid_mask = 1 - invalid_mask
    weight_matrix = weight_matrix * torch.unsqueeze(valid_mask, 0)

    return weight_matrix


def RW_full_weight(pc1, pc1_full, theta_2 = 0.8):
    distance2_matrix = torch.sum(pc1_full ** 2, 1, keepdim=True).transpose(2, 1) # B, N_full, 1
    distance2_matrix = distance2_matrix + torch.sum(pc1 ** 2, 1, keepdim=True)  # B, N_full, N
    distance2_matrix = distance2_matrix - 2 * torch.matmul(pc1_full.transpose(2, 1), pc1)

    weight_matrix = torch.exp(-distance2_matrix / theta_2)

    # If a point in pc1_full belongs to pc1, the element in identity_map is one; otherwise, zero.
    identity_map = distance2_matrix < 5e-4
    identity_map = identity_map / (torch.sum(identity_map, -1, keepdim=True) + 1e-8)

    return weight_matrix, identity_map


def RW_module(pc1, pseudo_label, RW_step, valid_vector, pc1_full, is_infinite_step, Alpha = 0.8):
    """
    Pseudo label refinement via random walk.
    When pc1 is sampled data, after generating refined pseudo labels for pc1,
    we upsample the refined pseudo labels to yield pseudo labels for pc1_full by a random walk step.

    ----------
    Input:
        valid_vector: one indicates that the point belongs to the labeled set,
                      and zeros indicates that it belongs to the unlabeled set. [B, N]
    """

    # Affinity matrix
    W = RW_weight(pc1)

    # Set the affinity elements from unlabeled points to all points to zero.
    W = W * valid_vector.unsqueeze(1)

    # Transition matrix
    A = W / (torch.sum(W, -1, keepdim=True) + 1e-8)

    pseudo_label = pseudo_label.permute(0, 2, 1)
    tmp = pseudo_label.clone()

    if is_infinite_step: # Refinement with an infinite random walk step
        # Propagating on the undirected subgraph
        Identity = torch.eye(W.shape[-1]).unsqueeze(0).cuda()
        tmp = (1 - Alpha) * torch.matmul(torch.inverse(Identity - Alpha * A), tmp)

        # Propagating on the directed subgraph
        tmp = (1 - valid_vector.unsqueeze(-1)) * torch.matmul(A, tmp) + valid_vector.unsqueeze(-1) * tmp
    else:
        Alpha_vector = 1 - (1 - Alpha) * valid_vector.unsqueeze(2)
        One_minus_alpha_vector = (1 - Alpha) * valid_vector.unsqueeze(2)

        for index in range(RW_step):
            # One iteration of random walk refinements
            tmp = Alpha_vector * torch.matmul(A, tmp) + One_minus_alpha_vector * pseudo_label


    if pc1_full is not None:
        # Upsample refined pseudo labels by a random walk step.
        W_full, identity_map = RW_full_weight(pc1, pc1_full)

        W_full = W_full * valid_vector.unsqueeze(1)
        W_full = W_full / (torch.sum(W_full, -1, keepdim=True) + 1e-7)

        mask = torch.sum(identity_map, -1, keepdim=True)

        refined_pseudo_label = (1 - mask) * torch.matmul(W_full, tmp) + mask * torch.matmul(identity_map, tmp)
    else:
        refined_pseudo_label = tmp

    return refined_pseudo_label

