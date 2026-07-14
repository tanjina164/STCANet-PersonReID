from __future__ import print_function, absolute_import
import sys, os, torch, cv2, random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# পাথ সেটআপ
PROJECT_ROOT = '/kaggle/working/STCANet-PersonReID'
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'tools'))

import models, data_manager
import transforms.spatial_transforms as ST
import transforms.temporal_transforms as TT
from video_loader import VideoDataset
from eval_metrics import evaluate

def visualize_baseline(distmat, query_paths, gallery_paths, q_pids, g_pids, q_camids, g_camids, save_dir='/kaggle/working/viz_baseline'):
    os.makedirs(save_dir, exist_ok=True)
    # 
    for i in range(3): 
        q_idx = random.randint(0, distmat.shape[0]-1)
        order = np.argsort(distmat[q_idx])
        fig, axes = plt.subplots(1, 6, figsize=(20, 5))
        axes[0].imshow(cv2.cvtColor(cv2.imread(query_paths[q_idx][0]), cv2.COLOR_BGR2RGB))
        axes[0].set_title("Query"); axes[0].axis('off')
        
        g_count, rank_idx = 0, 0
        while g_count < 5:
            g_idx = order[rank_idx]
            if g_pids[g_idx] == q_pids[q_idx] and g_camids[g_idx] == q_camids[q_idx]:
                rank_idx += 1; continue
            g_img = cv2.imread(gallery_paths[g_idx][0])
            color = (0, 255, 0) if g_pids[g_idx] == q_pids[q_idx] else (255, 0, 0)
            g_img = cv2.copyMakeBorder(g_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=color)
            axes[g_count+1].imshow(cv2.cvtColor(g_img, cv2.COLOR_BGR2RGB))
            axes[g_count+1].set_title(f"Rank-{g_count+1}"); axes[g_count+1].axis('off')
            g_count += 1; rank_idx += 1
        plt.savefig(os.path.join(save_dir, f'baseline_{i}.png')); plt.close()

def main():
    dataset = data_manager.init_dataset('market1501', root='/kaggle/input/datasets/jiniyatanjina')
    model = models.init_model('STCANet3D', num_classes=dataset.num_train_pids, use_gpu=True)
    checkpoint = torch.load('/kaggle/working/STCANet-PersonReID/logs/best_model_perfect.pth.tar', weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = torch.nn.DataParallel(model).cuda().eval()

    # ট্রান্সফর্মস কনফিগারেশন
    s_trans = ST.Compose([ST.Scale((256, 128)), ST.ToTensor(), ST.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t_trans = TT.TemporalBeginCrop(size=4)

    def get_feats(data_list):
        loader = DataLoader(VideoDataset(data_list, spatial_transform=s_trans, temporal_transform=t_trans), batch_size=1, shuffle=False)
        f, p, c = [], [], []
        with torch.no_grad():
            for v, pid, cid in loader:
                feat = model.module.bn(model(v.cuda()))
                f.append(feat.data.cpu()); p.extend(pid); c.extend(cid)
        return torch.cat(f, 0).numpy(), np.array(p), np.array(c)

    qf, q_pids, q_cam = get_feats(dataset.query)
    gf, g_pids, g_cam = get_feats(dataset.gallery)

    # ইউক্লিডিয়ান ডিস্ট্যান্স
    distmat = np.sum(qf**2, axis=1, keepdims=True) + np.sum(gf**2, axis=1, keepdims=True).T - 2 * np.dot(qf, gf.T)
    
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_cam, g_cam)
    
    print(f"--- Baseline Results (No Re-ranking) ---")
    print(f"mAP: {mAP:.2%}\nRank-1: {cmc[0]:.2%}")
    visualize_baseline(distmat, dataset.query, dataset.gallery, q_pids, g_pids, q_cam, g_cam)

if __name__ == '__main__': main()
