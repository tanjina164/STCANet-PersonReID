import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import importlib.util
from torchvision import transforms as T

# ১. ডাইনামিক রুট পাথ সেটআপ
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tools'))
sys.path.append(os.path.join(ROOT_DIR, 'data_manager'))

import data_manager
from video_loader import VideoDataset
from models.STCANet3D import STCANet3D

# utils/rerank.py থেকে অরিজিনাল ফাংশন 'k_reciprocal_re_ranking' লোড করা
rerank_path = os.path.join(ROOT_DIR, 'utils', 'rerank.py')
spec = importlib.util.spec_from_file_location("rerank", rerank_path)
rerank_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rerank_module)
re_ranking_fn = rerank_module.k_reciprocal_re_ranking

# evaluation.py থেকে অরিজিনাল মেট্রিক ক্যালকুলেটর 'evaluate' লোড করা
eval_path = os.path.join(ROOT_DIR, 'evaluation.py')
spec_eval = importlib.util.spec_from_file_location("evaluation", eval_path)
eval_module = importlib.util.module_from_spec(spec_eval)
spec_eval.loader.exec_module(eval_module)
evaluate_fn = eval_module.evaluate

def extract_features(model, dataloader):
    model.eval()
    features, pids, camids = [], [], []
    print("   -> Starting feature extraction loop...")
    with torch.no_grad():
        for batch_idx, (imgs, pid, camid) in enumerate(dataloader):
            # STCANet3D এর জন্য ইমেজকে ৫ডি টেনসরে রূপান্তর করা [B, C, T, H, W]
            if imgs.dim() == 4:
                imgs = imgs.unsqueeze(2) # [B, C, 1, H, W]
                
            imgs = imgs.cuda()
            outputs = model(imgs)
            
            # মেমোরি গ্রাফ বিচ্ছিন্ন করে নিরাপদভাবে NumPy-তে কনভার্ট করা
            features.append(outputs.detach().cpu().numpy())
            pids.extend(pid.numpy())
            camids.extend(camid.numpy())
            if (batch_idx + 1) % 50 == 0:
                print(f"      Batch {batch_idx + 1}/{len(dataloader)} processed.")
    return np.concatenate(features, axis=0), np.asarray(pids), np.asarray(camids)

def main():
    print("==> Loading STCANet3D Model...")
    # 🌟 পজিশনাল ও কি-ওয়ার্ড আর্গুমেন্ট ট্র্যাপ এড়াতে এক্সপ্লিসিটলি পাস করা হলো
    model = STCANet3D(num_classes=751, use_gpu=True, loss={'xent', 'htri'}).cuda()
    
    checkpoint_path = os.path.join(ROOT_DIR, 'logs', 'best_model.pth.tar')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model weights not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("==> Model weights loaded successfully!")

    print("==> Loading Market-1501 dataset...")
    dataset = data_manager.init_dataset(name='market1501', root='/kaggle/input/datasets/jiniyatanjina/')
    print("==> Market-1501 dataset loaded successfully!")

    standard_transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    queryloader = DataLoader(
        VideoDataset(dataset.query, spatial_transform=standard_transform, temporal_transform=None),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, spatial_transform=standard_transform, temporal_transform=None),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    print("==> DataLoaders successfully created!")

    print("==> Extracting Query features...")
    q_feat, q_pids, q_camids = extract_features(model, queryloader)
    
    print("==> Extracting Gallery features...")
    g_feat, g_pids, g_camids = extract_features(model, galleryloader)

    print("==> Computing original distance matrices...")
    q_g_dist = np.power(q_feat, 2).sum(axis=1, keepdims=True) + np.power(g_feat, 2).sum(axis=1, keepdims=True).T - 2 * np.dot(q_feat, g_feat.T)
    q_g_dist = np.nan_to_num(np.clip(q_g_dist, 0, None))

    q_q_dist = np.power(q_feat, 2).sum(axis=1, keepdims=True) + np.power(q_feat, 2).sum(axis=1, keepdims=True).T - 2 * np.dot(q_feat, q_feat.T)
    q_q_dist = np.nan_to_num(np.clip(q_q_dist, 0, None))

    g_g_dist = np.power(g_feat, 2).sum(axis=1, keepdims=True) + np.power(g_feat, 2).sum(axis=1, keepdims=True).T - 2 * np.dot(g_feat, g_feat.T)
    g_g_dist = np.nan_to_num(np.clip(g_g_dist, 0, None))

    print("==> Applying K-reciprocal Re-ranking...")
    final_dist = re_ranking_fn(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)
    print("==> [SUCCESS] Re-ranking completed successfully!")
    
    output_npy_path = os.path.join(ROOT_DIR, 'final_dist.npy')
    np.save(output_npy_path, final_dist)
    print(f"==> Saved final distance matrix to '{output_npy_path}'")

    # --- মেট্রিক ইভালুয়েশন ও নিরাপদ আউটপুট প্রিন্টিং ---
    print("\n==> Evaluating Re-ID performance...")
    q_pids_eval = np.array([x[1] for x in queryloader.dataset.dataset])
    q_camids_eval = np.array([x[2] for x in queryloader.dataset.dataset])
    g_pids_eval = np.array([x[1] for x in galleryloader.dataset.dataset])
    g_camids_eval = np.array([x[2] for x in galleryloader.dataset.dataset])

    cmc, mAP = evaluate_fn(final_dist, q_pids_eval, g_pids_eval, q_camids_eval, g_camids_eval)

    mAP_val = float(np.mean(mAP)) if hasattr(mAP, '__len__') else float(mAP)
    
    print("\n" + "="*40)
    print("        STCANet PipeLine Results       ")
    print("="*40)
    print(f"  mAP     : {mAP_val*100:.2f}%")
    
    ranks = [1, 5, 10, 20]
    for r in ranks:
        if len(cmc) >= r:
            rank_val = float(cmc[r-1])
            print(f"  Rank-{r:<3}: {rank_val*100:.2f}%")
            
    print("="*40)
    return

if __name__ == '__main__':
    main()
