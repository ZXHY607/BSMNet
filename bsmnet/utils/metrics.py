# import torch
# import torch.nn.functional as F
# from utils.experiment import make_nograd_func
# from torch.autograd import Variable
# from torch import Tensor


# # Update D1 from >3px to >=3px & >5%
# # matlab code:
# # E = abs(D_gt - D_est);
# # n_err = length(find(D_gt > 0 & E > tau(1) & E. / abs(D_gt) > tau(2)));
# # n_total = length(find(D_gt > 0));
# # d_err = n_err / n_total;

# def check_shape_for_metric_computation(*vars):
#     assert isinstance(vars, tuple)
#     for var in vars:
#         assert len(var.size()) == 3
#         assert var.size() == vars[0].size()

# # a wrapper to compute metrics for each image individually
# def compute_metric_for_each_image(metric_func):
#     def wrapper(D_ests, D_gts, masks, *nargs):
#         check_shape_for_metric_computation(D_ests, D_gts, masks)
#         bn = D_gts.shape[0]  # batch size
#         results = []  # a list to store results for each image
#         # compute result one by one
#         for idx in range(bn):
#             # if tensor, then pick idx, else pass the same value
#             cur_nargs = [x[idx] if isinstance(x, (Tensor, Variable)) else x for x in nargs]
#             if masks[idx].float().mean() / (D_gts[idx] > 0).float().mean() < 0.1:
#                 print("masks[idx].float().mean() too small, skip")
#             else:
#                 ret = metric_func(D_ests[idx], D_gts[idx], masks[idx], *cur_nargs)
#                 results.append(ret)
#         if len(results) == 0:
#             print("masks[idx].float().mean() too small for all images in this batch, return 0")
#             return torch.tensor(0, dtype=torch.float32, device=D_gts.device)
#         else:
#             return torch.stack(results).mean()
#     return wrapper

# @make_nograd_func
# @compute_metric_for_each_image
# def D1_metric(D_est, D_gt, mask):
#     D_est, D_gt = D_est[mask], D_gt[mask]
#     E = torch.abs(D_gt - D_est)
#     err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
#     return torch.mean(err_mask.float())

# @make_nograd_func
# @compute_metric_for_each_image
# def Thres_metric(D_est, D_gt, mask, thres):
#     assert isinstance(thres, (int, float))
#     D_est, D_gt = D_est[mask], D_gt[mask]
#     E = torch.abs(D_gt - D_est)
#     err_mask = E > thres
#     return torch.mean(err_mask.float())

# # NOTE: please do not use this to build up training loss
# @make_nograd_func
# # @compute_metric_for_each_image
# def EPE_metric(D_est, D_gt, mask):
#     D_est, D_gt = D_est[mask], D_gt[mask]

#     if mask.sum()==0:
#         return 0.0
#     # return F.l1_loss(D_est, D_gt, size_average=True)
#     else:
#         return F.l1_loss(D_est, D_gt, size_average=True)

# @make_nograd_func
# def Depth_metric(D_est, D_gt, mask):
#     D_est, D_gt = D_est[mask], D_gt[mask]

#     if mask.sum()==0:
#         return 0.0
#     # return F.l1_loss(D_est, D_gt, size_average=True)
#     else:
#         return F.l1_loss(D_est, D_gt, size_average=True)


# @make_nograd_func
# def xDepth_metric(D_est, D_gt, mask):
#     D_est, D_gt = D_est[mask], D_gt[mask]

#     if mask.sum()==0:
#         return [0.0], 0.0
#     # return F.l1_loss(D_est, D_gt, size_average=True)
#     else:
#         return [F.l1_loss(D_est, D_gt, size_average=True)], ((D_est- D_gt).abs()/D_gt).mean()

# @make_nograd_func
# def xD1_metric(D_est, D_gt, mask, rate):
#     D_est, D_gt = D_est[mask], D_gt[mask]
#     E = torch.abs(D_gt - D_est)
#     err_mask = (E / D_gt.abs() > rate)
#     return torch.mean(err_mask.float())

import torch
import torch.nn.functional as F
from utils.experiment import make_nograd_func
from torch.autograd import Variable
from torch import Tensor

def D1_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(err_mask.float())

def Thres_metric(D_est, D_gt, mask, thres):
    assert isinstance(thres, (int, float))
    D_est, D_gt = D_est[mask], D_gt[mask]
    E = torch.abs(D_gt - D_est)
    err_mask = E > thres
    return torch.mean(err_mask.float())

def EPE_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]
    return F.l1_loss(D_est, D_gt, reduction='mean')

def RelErr_metric(D_est, D_gt, mask=None):
    if mask is not None:
        D_est, D_gt = D_est[mask], D_gt[mask]

    log_est, log_gt = D_est.log10(), D_gt.log10()

    D_abs= (D_est - D_gt).abs()
    D_abs_2 = D_abs.pow(2)
    D_log = (log_est - log_gt).abs()
    
    absRel = (D_abs / D_gt).mean()
    # logmae = D_log.mean(0)
    sqRel = (D_abs_2 / D_gt).mean()
    rmse = torch.sqrt(D_abs_2.mean())
    logRMSE = torch.sqrt(D_log.pow(2).mean())

    return absRel, sqRel, rmse, logRMSE

def Accuracy_metric(D_est, D_gt, mask=None):
    if mask is not None:
        D_est, D_gt = D_est[mask], D_gt[mask]

    delta = torch.max(D_gt/D_est, D_est/D_gt)
    delta1 = (delta < 1.25).float().mean()
    delta2 = (delta < 1.25**2).float().mean()
    delta3 = (delta < 1.25**3).float().mean()

    return delta1, delta2, delta3

def Outlier_metric(D_est, D_gt, rate, mask=None):
    if mask is not None:
        D_est, D_gt = D_est[mask], D_gt[mask]

    E = torch.abs(D_gt - D_est)
    err_mask = (E / D_gt.abs() > rate)
    return torch.mean(err_mask.float())

def Depth_metric(D_est, D_gt, mask):
    D_est, D_gt = D_est[mask], D_gt[mask]

    if mask.sum()==0:
        return 0.0
    # return F.l1_loss(D_est, D_gt, size_average=True)
    else:
        return F.l1_loss(D_est, D_gt, size_average=True)