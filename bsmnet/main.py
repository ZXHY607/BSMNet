from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss
from utils import *
from torch.utils.data import DataLoader
import gc
import cv2
import torchvision
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-g', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdepth', type=int, default=80, help='maximum disparity')
parser.add_argument('--baseline', type=float, default=0.54, help='baseline')
parser.add_argument('--focus', type=float, default=1003., help='focus distance')

parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=20, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--save_path', type=str, default='error-all/',
                    help='the path of saving checkpoints and log')

# parse arguments, set seeds
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

# create summary logger
print("creating new summary file")
logger = SummaryWriter(args.logdir)

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True)
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdepth)
model = nn.DataParallel(model)
model.cuda()

for index, (name, value) in enumerate(model.named_parameters()):
    value.requires_grad = True
    # if ((index >= 204)& (index <= 218)): #scale attention
    # if ((index >= 204) & (index <= 210)): #stop scale
    # if ((index >= 211) & (index <= 218)): #stop attention
        # value.requires_grad = False
    # print(index, name, " : ", value.requires_grad)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

# load parameters
start_epoch = 0
if args.resume:
    # find all checkpoints file and sort according to epoch id
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load the checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.load_state_dict(state_dict['model'])
    start_epoch = state_dict['epoch'] + 1
print("start at epoch {}".format(start_epoch))


def train():
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        # lr_scheduler.step()
        # training
        # TrainImgLoader=[]
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss,
                                                                                       time.time() - start_time))
        # saving checkpoints
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0

            loss, scalar_outputs, image_outputs = test_sample(batch_idx, sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                                     batch_idx,
                                                                                     len(TestImgLoader), loss,
                                                                                     time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, depth_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    depth_gt = depth_gt.cuda()

    optimizer.zero_grad()

    depth_ests,_ = model(imgL, imgR, depth_gt, args.baseline, args.focus)

    scale = 1.0

    disparity_est = [args.baseline * args.focus / (disp_est + 0.00000001) for disp_est in depth_ests]
    disp_gt = args.baseline * args.focus / (depth_gt + 0.00000001)

    mask_dis = (disp_gt < args.baseline * args.focus / 1.) & (disp_gt > args.baseline * args.focus / 200.)
    mask = (depth_gt <= args.maxdepth) & (depth_gt > 1.0)

    loss = model_loss(depth_ests, depth_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_ests, "depth_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap_dis"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disparity_est]
            image_outputs["errormap_depth"] = [disp_error_image_func(depth_est, depth_gt) for depth_est in depth_ests]

            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask_dis) for disp_est in disparity_est]
            scalar_outputs["Depth"] = [Depth_metric(depth_est, depth_gt, mask) for depth_est in depth_ests]

            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask_dis) for disp_est in disparity_est]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask_dis, 1.0) for disp_est in disparity_est]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask_dis, 2.0) for disp_est in disparity_est]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask_dis, 3.0) for disp_est in disparity_est]
    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(batch_idx, sample, compute_metrics=True):
    model.eval()

    imgL, imgR, depth_gt = sample['left'], sample['right'], sample['disparity']

    imgL = imgL.cuda()
    imgR = imgR.cuda()
    depth_gt = depth_gt.cuda()
    te = depth_gt - depth_gt
    # start_time = time.time()
    depth_ests,sca = model(imgL, imgR, depth_gt, args.baseline, args.focus)
    # torch.cuda.synchronize()
    # t=time.time() - start_time
    # print('time-one',time.time() - start_time)
    scale = 1.0

    disparity_est = [args.baseline * args.focus / (disp_est + 0.00000001) for disp_est in depth_ests]
    disp_gt = args.baseline * args.focus / (depth_gt + 0.00000001)

    # mask_dis = (disp_gt < args.baseline * args.focus / 1.) & (disp_gt > args.baseline * args.focus / 100.)
    # mask = (depth_gt <= args.maxdepth) & (depth_gt > 1.0)
    mask =  (disp_gt > 0)&(disp_gt < 192) & (depth_gt <= args.maxdepth) & (depth_gt > 1.0)
    loss = model_loss(depth_ests, depth_gt, mask)

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_ests, "depth_gt": disp_gt, "imgL": imgL, "imgR": imgR}

    # if compute_metrics:
    #     with torch.no_grad():
    image_outputs["errormap_dis"] = [disp_error_image_func(disp_est, disp_gt) for disp_est in disparity_est]
    image_outputs["errormap_depth"] = [disp_error_image_func(depth_est, depth_gt) for depth_est in depth_ests]

    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disparity_est]
    scalar_outputs["Depth"] = [Depth_metric(depth_est, depth_gt, mask) for depth_est in depth_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disparity_est]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disparity_est]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disparity_est]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disparity_est]

    depth_gt[~mask] = 0
    for depth_est in depth_ests:
        depth_est[~mask] = 0
    
    depth_ests = depth_ests[-1]
    
    #
    ma = depth_ests > 1000
    depth_ests[ma] = 1000.0
    
    
    mask = (depth_gt >= 100)
    scalar_outputs["depth100"] = Depth_metric(depth_ests, depth_gt, mask)
    
    mask = (depth_gt < 100) & (depth_gt >= 90)
    scalar_outputs["depth90"] = Depth_metric(depth_ests, depth_gt, mask)
    
    mask = (depth_gt < 90) & (depth_gt >= 80)
    scalar_outputs["depth80"] = Depth_metric(depth_ests, depth_gt, mask)
    
    mask = (depth_gt < 80) & (depth_gt >= 70)
    scalar_outputs["depth70"] = Depth_metric(depth_ests, depth_gt, mask)
    
    mask = (depth_gt < 70) & (depth_gt >= 60)
    scalar_outputs["depth60"] = Depth_metric(depth_ests, depth_gt, mask)
    
    mask = (depth_gt < 60) & (depth_gt >= 50)
    scalar_outputs["depth50"] = Depth_metric(depth_ests, depth_gt, mask)
    
    mask = (depth_gt < 50) & (depth_gt >= 40)
    scalar_outputs["depth40"] = Depth_metric(depth_ests, depth_gt, mask)
    
    mask = (depth_gt < 40) & (depth_gt >= 30)
    scalar_outputs["depth30"] = Depth_metric(depth_ests, depth_gt, mask)
    
    mask = (depth_gt < 30) & (depth_gt >= 20)
    scalar_outputs["depth20"] = Depth_metric(depth_ests, depth_gt, mask)
    
    mask = (depth_gt < 20) & (depth_gt >= 10)
    scalar_outputs["depth10"] = Depth_metric(depth_ests, depth_gt, mask)
    
    mask = (depth_gt < 10) & (depth_gt > 1)
    scalar_outputs["depth0"] = Depth_metric(depth_gt, depth_ests, mask)

    # if compute_metrics:
    #     image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]

    # error_our = cv2.imread(join('/data/zh/accuracy/Gwc_multimatch_color_sub/error/', "errorim-%04d.png" % batch_idx),-1)/256
    #
    # error_G = torch.squeeze((disp_gt-disp_ests[0]).abs())
    #
    # heat = np.array(error_G.cpu())-error_our
    # mask_heat = heat<0
    # heat[mask_heat]=0
    # print(imgL)

    # L_im=torch.squeeze(imgL).permute(1,2,0)
    # heatmap = cv2.applyColorMap(np.uint8(torch.squeeze(heat,0).cpu()), cv2.COLORMAP_JET)  # 利用色彩空间转换将heatmap凸显
    # heatmap = np.float32(heatmap) / 255  # 归一化
    # # print(heat.shape,heatmap.shape,L_im.shape)
    # cam = heatmap + np.float32(np.array(L_im.cpu()))  # 将heatmap 叠加到原图
    # cam = cam / np.max(cam)
    # cv2.imwrite(join(args.save_path, "heat-%06d.jpg" % batch_idx), np.uint8(255*cam))  # 生成图像

    # 1050，27
    # disp_gt[1 - mask] = 0
    # disp_ests[0][1-mask] = 0
    # disp_gt[mask] = 721.5 * 0.54 / disp_gt[mask]
    # disp_ests[0][mask] = 721.5 * 0.54 / disp_ests[0][mask]
    # image_error = torch.squeeze((disp_ests[0] - disp_gt).abs())
    # mm=image_error>255
    # image_error[mm]=255
    # image_error = 256 * image_error
    # cv2.imwrite(join(args.save_path, "errorim-%04d.png" % batch_idx), np.array(image_error.cpu()).astype(np.uint16))

    # te = torch.zeros_like(disp_gt)
    # # disp_ests[1]=disp_ests[1]*255/ (disp_ests[1].max()-disp_ests[1].min())
    # print(disp_ests[1])
    # image_error = disp_error_image_func(te, torch.squeeze(sca, 1))
    # torchvision.utils.save_image(image_error, join(args.save_path, "sca-%06d.jpg" % batch_idx))
    # image_error = disp_error_image_func(depth_ests[0], depth_gt)
    # torchvision.utils.save_image(image_error, join(args.save_path, "pred0-%06d.jpg" % batch_idx))
    # image_error = disp_error_image_func(depth_ests[1], depth_gt)
    # torchvision.utils.save_image(image_error, join(args.save_path, "pred1-%06d.jpg" % batch_idx))

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    train()
