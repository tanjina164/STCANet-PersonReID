from __future__ import print_function, absolute_import
import os
import gc
import sys
import time
import math
import h5py
import scipy
import datetime
import argparse
import os.path as osp
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
sys.path.append('tools/')

import data_manager
from video_loader import VideoDataset
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
import models

from utils import AverageMeter, Logger, save_checkpoint
from evaluation import evaluation

parser = argparse.ArgumentParser(description='Train video model with cross entropy loss')
# Datasets
parser.add_argument('--root', type=str, default='')
parser.add_argument('-d', '--dataset', type=str, default='duke',
                    choices=data_manager.get_names())
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--height', type=int, default=256,
                    help="height of an image (default: 256)")
parser.add_argument('--width', type=int, default=128,
                    help="width of an image (default: 128)")
# Augment
parser.add_argument('--seq_len', type=int, default=4, help="number of images to sample in a tracklet")
parser.add_argument('--sample_stride', type=int, default=8, help="stride of images to sample in a tracklet")
parser.add_argument('--test_frames', default=4, type=int, help='frames/clip for test')
# Optimization options
parser.add_argument('--distance', type=str, default='consine', help="euclidean or consine")
# Architecture
parser.add_argument('-a', '--arch', type=str, default='STCANet3D')
parser.add_argument('--save-dir', type=str, default='')
parser.add_argument('--resume', type=str, default='', metavar='PATH')
# Miscs
parser.add_argument('--seed', type=int, default=1, help="manual seed")
parser.add_argument('--use_cpu', action='store_true', help="use cpu")
parser.add_argument('--gpu_devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

def safe_load_state_dict(model, checkpoint_path, device):
    """
    নিরাপদ এবং ডাইনামিক চেকপয়েন্ট লোডার।
    এটি weights_only এবং শেইপ মিসম্যাচজনিত লেয়ারগুলোকে ফিল্টার করে লোড নিশ্চিত করে।
    """
    print("Loading checkpoint from '{}'".format(checkpoint_path))
    # PyTorch 2.6+ সিকিউরিটি বাইপাস এবং জিপিইউ ম্যাপিং
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['state_dict']
    model_dict = model.state_dict()
    
    # শেইপ ম্যাচিং এবং আনম্যাচড লেয়ার ফিল্টারিং
    filtered_dict = {}
    for k, v in state_dict.items():
        if k in model_dict:
            if v.shape == model_dict[k].shape:
                filtered_dict[k] = v
            else:
                print("[WARNING] Size mismatch for {}: checkpoint {} vs model {}. Skipped!".format(k, v.shape, model_dict[k].shape))
        else:
            print("[INFO] Hidden/Missing key in model structure: {}. Skipped!".format(k))
            
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print("[SUCCESS] Checkpoint successfully loaded!")

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_test1.txt'), mode='a')
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu_devices))
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset, root=args.root)

    # Data augmentation
    spatial_transform_test = ST.Compose([
                ST.Scale((args.height, args.width), interpolation=3),
                ST.ToTensor(),
                ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    # Testing এ ৩ডি মডেল ডাইমেনশন ঠিক রাখতে seq_len কে টেম্পোরাল ট্রান্সফর্ম দেওয়া হলো
    temporal_transform_test = TT.TemporalBeginCrop(size=args.test_frames)

    pin_memory = True if use_gpu else False

    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=1, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=1, shuffle=False, num_workers=0,
        pin_memory=pin_memory, drop_last=False
    )

    print("Initializing model: {}".format(args.arch))
    model = models.init_model(name=args.arch, use_gpu=use_gpu, num_classes=dataset.num_train_pids, loss={'xent', 'htri'})

    if use_gpu:
        model = model.cuda()
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    # সিঙ্গেল মডেল লোডিং বা রেজুমিং
    if args.resume:
        safe_load_state_dict(model, args.resume, device)
        test_all_frames(model, queryloader, galleryloader, use_gpu)

    # মাল্টিপল বা লাস্ট ট্রেইনড ওয়েইট চেকপয়েন্ট লোডিং
    if not args.resume:
        for epoch in [150, 140, 130, 120]:
            weights = os.path.join(args.save_dir, 'checkpoint_ep'+str(epoch)+'.pth.tar')
            if not os.path.isfile(weights):
                continue
            safe_load_state_dict(model, weights, device)
            test_all_frames(model, queryloader, galleryloader, use_gpu)

def test_all_frames(model, queryloader, galleryloader, use_gpu):
    if use_gpu:
        model = nn.DataParallel(model)
    model.eval()
    with torch.no_grad():
        evaluation(model, args, queryloader, galleryloader, use_gpu)

if __name__ == '__main__':
    main()
