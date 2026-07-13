import sys
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# প্রজেক্ট পাথ যুক্ত করা
sys.path.append('/kaggle/working/Person-ReID-STCANet')

import data_manager
from video_loader import VideoDataset
import transforms.spatial_transforms as ST
from models.STCANet3D import STCANet3D
from utils.rerank import k_reciprocal_re_ranking

def extract_features(model, dataloader):
    model.eval()
    features, pids, camids = [], [], []
    print("   -> Starting feature extraction loop...")
    with torch.no_grad():
        for batch_idx, (imgs, pid, camid) in enumerate(dataloader):
            imgs = imgs.cuda()
            outputs = model(imgs)
            features.append(outputs.cpu().numpy())
            pids.extend(pid.numpy())
            camids.extend(camid.numpy())
            if (batch_idx + 1) % 20 == 0:
                print(f"      Batch {batch_idx + 1}/{len(dataloader)} processed.")
    return np.concatenate(features, axis=0), np.asarray(pids), np.asarray(camids)

def main():
    # ১. মডেল লোড করা
    print("==> Loading STCANet3D Model...")
    model = STCANet3D(num_classes=751, use_gpu=True).cuda()
    checkpoint = torch.load('/kaggle/working/Person-ReID-STCANet/logs/best_model.pth.tar', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    print("==> Model weights loaded successfully!")

    # ২. ডেটাসেট ইনিশিয়ালাইজেশন (Market-1501)
    print("==> Loading Market-1501 dataset...")
    dataset = data_manager.init_dataset(name='market1501', root='/kaggle/input/')

    # ৩. ইমেজ ট্রান্সফর্মেশন
    spatial_transform = ST.Compose([
        ST.Resize((256, 128)),
        ST.ToTensor(),
        ST.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ৪. কুয়েরি এবং গ্যালারি লোডার তৈরি করা
    queryloader = DataLoader(
        VideoDataset(dataset.query, seq_len=4, sample='video_train', transform=spatial_transform),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    galleryloader = DataLoader(
        VideoDataset(dataset.gallery, seq_len=4, sample='video_train', transform=spatial_transform),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    print("==> DataLoaders successfully created!")

    # ৫. ফিচার এক্সট্রাকশন
    print("==> Extracting Query features...")
    q_feat, q_pids, q_camids = extract_features(model, queryloader)
    
    print("==> Extracting Gallery features...")
    g_feat, g_pids, g_camids = extract_features(model, galleryloader)

    # ৬. অরিজিনাল ডিস্ট্যান্স ম্যাট্রিক্স গণনা
    print("==> Computing original distance matrices...")
    q_g_dist = np.power(q_feat, 2).sum(axis=1, keepdims=True) + np.power(g_feat, 2).sum(axis=1, keepdims=True).T - 2 * np.dot(q_feat, g_feat.T)
    q_g_dist = np.nan_to_num(np.clip(q_g_dist, 0, None))

    q_q_dist = np.power(q_feat, 2).sum(axis=1, keepdims=True) + np.power(q_feat, 2).sum(axis=1, keepdims=True).T - 2 * np.dot(q_feat, q_feat.T)
    q_q_dist = np.nan_to_num(np.clip(q_q_dist, 0, None))

    g_g_dist = np.power(g_feat, 2).sum(axis=1, keepdims=True) + np.power(g_feat, 2).sum(axis=1, keepdims=True).T - 2 * np.dot(g_feat, g_feat.T)
    g_g_dist = np.nan_to_num(np.clip(g_g_dist, 0, None))

    # ৭. রি-র‍্যাংকিং অ্যালগরিদম ট্রিগার করা
    print("==> Applying K-reciprocal Re-ranking...")
    final_dist = k_reciprocal_re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)

    print("==> [SUCCESS] Re-ranking completed successfully in standalone script!")
    
    # ফাইনাল ডিস্ট্যান্স ম্যাট্রিক্স ফিউচার ইউজের জন্য সেভ করে রাখা (ঐচ্ছিক)
    np.save('/kaggle/working/Person-ReID-STCANet/final_dist.npy', final_dist)
    print("==> Saved final distance matrix to 'final_dist.npy'")

if __name__ == '__main__':
    main()
