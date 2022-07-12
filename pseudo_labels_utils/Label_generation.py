from .OT_RW_utils import OT_RW


def FT3D_label_generation(pred_flows, pc1, pc2, norm1, norm2):

    # sample 2048 points as seeds to produce pseudo labels
    # and generate dense pseudo labels for all points by random walk refinement
    sub_pc1 = pc1[:, :2048, :].transpose(2, 1).contiguous()
    sub_pc2 = pc2[:, :2048, :].transpose(2, 1).contiguous()
    sub_norm1 = norm1[:, :2048, :].transpose(2, 1).contiguous()
    sub_norm2 = norm2[:, :2048, :].transpose(2, 1).contiguous()

    sub_pred_flows = pred_flows[:, :, :2048].contiguous()

    pseudo_gt = OT_RW(sub_pc1, sub_pc2, sub_pred_flows, sub_norm1, sub_norm2, None, None,
                      pc1_full=pc1.transpose(2, 1).contiguous(), OT_iter=4,
                      RW_step=5, is_infinite_step=False, Alpha=0.8)

    return pseudo_gt


def KITTI_label_generation(pred_flows, pc1, pc2, norm1, norm2, color1, color2):

    pc1 = pc1.transpose(2, 1).contiguous()
    pc2 = pc2.transpose(2, 1).contiguous()
    norm1 = norm1.transpose(2, 1).contiguous()
    norm2 = norm2.transpose(2, 1).contiguous()
    color1 = color1.transpose(2, 1).contiguous()
    color2 = color2.transpose(2, 1).contiguous()

    pseudo_gt = OT_RW(pc1, pc2, pred_flows, norm1, norm2, color1, color2,
                      pc1_full=None, OT_iter=4,
                      RW_step=None, is_infinite_step=True, Alpha=0.80)

    return pseudo_gt
