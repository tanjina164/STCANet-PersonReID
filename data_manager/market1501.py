from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import glob
import re

class Market1501(object):
    def __init__(self, root='/kaggle/working/', **kwargs):
        self.dataset_dir = osp.join(root, 'market1501', 'Market-1501-v15.09.15')
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        train, num_train_pids, num_train_tracklets = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_tracklets = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_tracklets = self._process_dir(self.gallery_dir, relabel=False)

        print("=> Market1501 Loaded as Pseudo-Video Tracklets")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d-]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            filename = osp.basename(img_path)
            if filename.startswith('-1') or filename.startswith('0000'):
                continue
            pid, _ = map(int, pattern.search(filename).groups())
            pid_container.add(pid)
        
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        tracklets = []
        for img_path in img_paths:
            filename = osp.basename(img_path)
            if filename.startswith('-1') or filename.startswith('0000'):
                continue
            
            pid, camid = map(int, pattern.search(filename).groups())
            assert 1 <= camid <= 6
            
            if relabel:
                pid = pid2label[pid]
            camid -= 1
            
            img_paths_tuple = (img_path,)
            tracklets.append((img_paths_tuple, pid, camid))

        return tracklets, len(pid_container), len(tracklets)
