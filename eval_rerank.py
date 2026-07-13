import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.append('/kaggle/working/Person-ReID-STCANet')
from models.STCANet3D import STCANet3D
from utils.rerank import k_reciprocal_re_ranking

def extract_features(model, dataloader):
    model.eval()
    features, pids, camids = [], [], []
    with torch.no_grad():
        for batch_idx, (imgs, pid, camid) in enumerate(dataloader):
            imgs = imgs.cuda()
            outputs = model(imgs)
            features.append(outputs.cpu().numpy())
            pids.extend(pid.numpy())
            camids.extend(camid.numpy())
    return np.concatenate(features, axis=0), np.asarray(pids), np.asarray(camids)

def main():
    print("--> Loading STCANet3D Model...")
    model = STCANet3D(num_classes=751, use_gpu=True).cuda()
    checkpoint = torch.load('/kaggle/working/Person-ReID-STCANet/logs/best_model.pth.tar', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("--> Model weights loaded successfully!")

    # নোটবুকের গ্লোবাল এনভায়রনমেন্ট থেকে লোডারগুলো অ্যাক্সেস করা হচ্ছে
    # (ফাইলটি যখন কাগল সেলে রান হবে তখন মেমরিতে থাকা লোডার রিড করবে)
    global queryloader, galleryloader
    
    print("--> Extracting Query and Gallery features...")
    q_feat, q_pids, q_camids = extract_features(model, queryloader)
    g_feat, g_pids, g_camids = extract_features(model, galleryloader)

    print("--> Computing original distance matrices...")
    q_g_dist = np.power(q_feat, 2).sum(axis=1, keepdims=True) + np.power(g_feat, 2).sum(axis=1, keepdims=True).T - 2 * np.dot(q_feat, g_feat.T)
    q_g_dist = np.nan_to_num(np.clip(q_g_dist, 0, None))

    q_q_dist = np.power(q_feat, 2).sum(axis=1, keepdims=True) + np.power(q_feat, 2).sum(axis=1, keepdims=True).T - 2 * np.dot(q_feat, q_feat.T)
    q_q_dist = np.nan_to_num(np.clip(q_q_dist, 0, None))

    g_g_dist = np.power(g_feat, 2).sum(axis=1, keepdims=True) + np.power(g_feat, 2).sum(axis=1, keepdims=True).T - 2 * np.dot(g_feat, g_feat.T)
    g_g_dist = np.nan_to_num(np.clip(g_g_dist, 0, None))

    print("--> Applying K-reciprocal Re-ranking...")
    final_dist = k_reciprocal_re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)
    
    print("--> Re-ranking completed inside script!")
    # এখানে তোমার ফাইনাল CMC/mAP ক্যালকুলেশন স্ক্রিপ্ট কল হতে পারে

if __name__ == '__main__':
    main()
