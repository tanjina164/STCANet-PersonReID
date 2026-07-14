import os
import sys

print("=== Starting Market-1501 LIP Masking Pipeline ===")

SHPLIP_DIR = "/kaggle/working/Single-Human-Parsing-LIP"

print("\n--> Step 1: Generating masks for bounding_box_train...")
os.system(f"python {SHPLIP_DIR}/inference.py /kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15/bounding_box_train /kaggle/working/market1501_masks/bounding_box_train --backend resnet50")

print("\n--> Step 2: Generating masks for bounding_box_test...")
os.system(f"python {SHPLIP_DIR}/inference.py /kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15/bounding_box_test /kaggle/working/market1501_masks/bounding_box_test --backend resnet50")

print("\n--> Step 3: Generating masks for query...")
os.system(f"python {SHPLIP_DIR}/inference.py /kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15/query /kaggle/working/market1501_masks/query --backend resnet50")

print("\n--> Step 4: Creating symlinks for unified dataset structure...")
os.makedirs("/kaggle/working/market1501/Market-1501-v15.09.15/", exist_ok=True)

os.system("ln -s /kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15/bounding_box_train /kaggle/working/market1501/Market-1501-v15.09.15/bounding_box_train")
os.system("ln -s /kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15/bounding_box_test /kaggle/working/market1501/Market-1501-v15.09.15/bounding_box_test")
os.system("ln -s /kaggle/input/datasets/jiniyatanjina/market1501/Market-1501-v15.09.15/query /kaggle/working/market1501/Market-1501-v15.09.15/query")

os.system("ln -s /kaggle/working/market1501_masks/bounding_box_train /kaggle/working/market1501/Market-1501-v15.09.15/bounding_box_train_masks")
os.system("ln -s /kaggle/working/market1501_masks/bounding_box_test /kaggle/working/market1501/Market-1501-v15.09.15/bounding_box_test_masks")
os.system("ln -s /kaggle/working/market1501_masks/query /kaggle/working/market1501/Market-1501-v15.09.15/query_masks")

print("\n[SUCCESS] Entire Masking and Linking pipeline completed successfully!")
