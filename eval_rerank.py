from __future__ import print_function, absolute_import
import sys
import os
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = '/kaggle/working/STCANet-PersonReID'
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'tools'))

import models
import data_manager
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
from video_loader import VideoDataset
from utils import Logger
from eval_metrics import evaluate

# 🎯 ১০০% মেমোরি-সেফ এবং ওরিজিনাল K-reciprocal রি-র‍্যাংকিং (ডট প্রোডাক্ট অপ্টিমাইজড)
def safe_k_reciprocal_re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    num_q = q_g_dist.shape[0]
    num_g = q_g_dist.shape[1]
    num_total = num_q + num_g
    
    # গোরোবাল ডিস্ট্যান্স ম্যাট্রিক্স
    original_dist = np.concatenate(
        [np.concatenate([q_q_dist, q_g_dist], axis=1),
         np.concatenate([q_g_dist.T, g_g_dist], axis=1)], axis=0
    )
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = original_dist / np.max(original_dist, axis=0)
    
    # ইনিশিয়াল র‍্যাংকিং
    initial_rank = np.argpartition(original_dist, range(1, k1 + 1), axis=1)
    
    nn_k1 = initial_rank[:, :k1]
    nn_k1_half = initial_rank[:, :int(np.ceil(k1 / 2))]
    
    V = np.zeros((num_total, num_total), dtype=np.float32)
    for i in range(num_total):
        k_reciprocal_index = nn_k1[i][np.isin(nn_k1[i], nn_k1[nn_k1[i]], assume_unique=True).reshape(nn_k1[i].shape)]
        
        candidates = nn_k1_half[i][~np.isin(nn_k1_half[i], k_reciprocal_index)]
        if len(candidates) > 0:
            for candidate in candidates:
                c_candidates = nn_k1[candidate][np.isin(nn_k1[candidate], nn_k1[nn_k1[candidate]])]
                if len(np.intersect1d(k_reciprocal_index, c_candidates)) > 2/3 * len(c_candidates):
                    k_reciprocal_index = np.append(k_reciprocal_index, candidate)
                    
        k_reciprocal_index = np.unique(k_reciprocal_index)
        weight = np.exp(-original_dist[i, k_reciprocal_index])
        V[i, k_reciprocal_index] = weight / np.sum(weight)
    
    original_dist = original_dist[:num_q, num_q:]
    if k2 > 1:
        # 💡 থ্রি-ডি ব্রডকাস্টিং বাদ দিয়ে টু-ডি ডট প্রোডাক্টের মাধ্যমে জ্যাকার্ড ডিস্ট্যান্স নির্ণয় (মেমোরি বাঁচানোর ম্যাজিক)
        V_q = V[:num_q, :]
        V_g = V[num_q:, :]
        
        # Intersection এবং Union হিসাব করা টু-ডি স্পেসে
        print("--> Calculating Jaccard matrix using safe dot-product...")
        V_g_T = V_g.T
        intersection = np.dot(V_q, V_g_T) # Shape: (num_q, num_g)
        
        # Union = sum(V_q) + sum(V_g) - intersection
        sum_q = np.sum(V_q, axis=1, keepdims=True) # Shape: (num_q, 1)
        sum_g = np.sum(V_g, axis=1, keepdims=True).T # Shape: (1, num_g)
        union = sum_q + sum_g - intersection
        
        re_rank_dist = 1.0 - (intersection / (union + 1e-12))
        return lambda_value * original_dist + (1 - lambda_value) * re_rank_dist
        
    return original_dist

parser = argparse.ArgumentParser(description='Evaluate STCANet3D with Memory-Safe Re-ranking')
parser.add_argument('--root', type=str, default='')
parser.add_argument('-d', '--dataset', type=str, default='market1501', choices=['mars', 'duke', 'market1501'])
parser.add_argument('-j', '--workers', default=2, type=int, help="number of data loading workers")
parser.add_argument('--height', type=int, default=256, help="height of image")
parser.add_argument('--width', type=int, default=128, help="width of image")
parser.add_argument('--test_frames', default=4, type=int, help='frames/clip for test')
parser.add_argument('--test_batch', default=1, type=int, help="batch size for testing (must be 1)")

parser.add_argument('-a', '--arch', type=str, default='STCANet3D')
parser.add_argument('--resume', type=str, default='', required=True, metavar='PATH')
parser.add_argument('--save-dir', type=str, default='')
parser.add_argument('--gpu_devices', default='0', type=str, help='gpu device ids')

parser.add_argument('--k1', type=int, default=20)
parser.add_argument('--k2', type=int, default=6)
parser.add_argument('--lambda_value', type=float, default=0.3)

args = parser.parse_args()

def extract_features(model, dataloader, use_gpu):
    model.eval()
    features, pids, camids = [], [], []
    with torch.no_grad():
        for batch_idx, (vids, pid, camid) in enumerate(dataloader):
            if use_gpu: vids = vids.cuda()
            feat = model(vids)
            feat = model.module.bn(feat) if isinstance(model, torch.nn.DataParallel) else model.bn(feat)
            features.append(feat.data.cpu())
            pids.extend(pid)
            camids.extend(camid)
    return torch.cat(features, 0), np.asarray(pids), np.asarray(camids)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        sys.stdout = Logger(os.path.join(args.save_dir, 'log_rerank.txt'), mode='a')

    print("Initializing dataset {}".format(args.dataset))
    dataset = data_manager.init_dataset(name=args.dataset, root=args.root)

    spatial_transform_test = ST.Compose([
        ST.Scale((args.height, args.width), interpolation=3),
        ST.ToTensor(),
        ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    temporal_transform_test = TT.TemporalBeginCrop(size=args.test_frames)

    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=use_gpu, drop_last=False
    )
    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=spatial_transform_test, temporal_transform=temporal_transform_test),
        batch_size=args.test_batch, shuffle=False, num_workers=args.workers, pin_memory=use_gpu, drop_last=False
    )

    model = models.init_model(name=args.arch, num_classes=dataset.num_train_pids, use_gpu=use_gpu, loss={'xent', 'htri'})
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    
    if use_gpu: model = torch.nn.DataParallel(model).cuda()

    since = time.time()
    print("--> Extracting Query and Gallery features...")
    qf, q_pids, q_camids = extract_features(model, queryloader, use_gpu)
    gf, g_pids, g_camids = extract_features(model, galleryloader, use_gpu)

    print("Extracting features complete in {:.0f}m {:.0f}s".format((time.time()-since)//60, (time.time()-since)%60))

    print("Computing squared Euclidean distance matrices...")
    qf, gf = qf.numpy(), gf.numpy()
    q_g_dist = np.clip(np.sum(np.power(qf, 2), axis=1, keepdims=True) + np.sum(np.power(gf, 2), axis=1, keepdims=True).T - 2 * np.dot(qf, gf.T), 0, None)
    q_q_dist = np.clip(np.sum(np.power(qf, 2), axis=1, keepdims=True) + np.sum(np.power(qf, 2), axis=1, keepdims=True).T - 2 * np.dot(qf, qf.T), 0, None)
    g_g_dist = np.clip(np.sum(np.power(gf, 2), axis=1, keepdims=True) + np.sum(np.power(gf, 2), axis=1, keepdims=True).T - 2 * np.dot(gf, gf.T), 0, None)

    print("--> Applying Safe Fast K-reciprocal Re-ranking...")
    rerank_since = time.time()
    final_dist = safe_k_reciprocal_re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=args.k1, k2=args.k2, lambda_value=args.lambda_value)
    print("Re-ranking complete in {:.0f}s".format(time.time() - rerank_since))

    print("Computing CMC and mAP...")
    cmc, mAP = evaluate(final_dist, q_pids, g_pids, q_camids, g_camids)

    print("\nResults with Re-ranking ----------")
    print("mAP: {:.2%}".format(mAP))
    print("CMC curve")
    for r in [1, 5, 10, 20]:
        print("Rank-{:<3}: {:.2%}".format(r, cmc[r-1]))
    print("----------------------------------")

if __name__ == '__main__':
    main()
