import torch.nn.functional as F


# def model_loss(disp_ests, disp_gt, mask):
#     weights = [0.5, 0.5, 0.7, 1.0]
#     all_losses = []
#     for disp_est, weight in zip(disp_ests, weights):
#         all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
#     return sum(all_losses)


def model_loss(disp_ests, disp_gt, mask):
    # weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []

    
    for disp_est in disp_ests:

        all_losses.append(((disp_gt[mask]-disp_est[mask]).abs()).mean())
    # if scale_ests is not None:
        # for scale in scale_ests:
            # all_losses.append(0.0*scale[mask].mean())
    return sum(all_losses)