import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from coperception.datasets import V2XSimDet
from coperception.configs import Config, ConfigGlobal

import logging
LOG = logging.getLogger(__name__)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train', debug=False):
        self.args = args
        self.split = split
        # self.sample_length = args['sample_length']
        # self.size = self.w, self.h = (args['w'], args['h'])

        # with open(os.path.join(args['data_root'], args['name'], split+'.json'), 'r') as f:
        #     self.video_dict = json.load(f)
        # self.video_names = list(self.video_dict.keys())
        # if debug or split != 'train':
        #     self.video_names = self.video_names[:100]

        # self._to_tensors = transforms.Compose([
        #     Stack(),
        #     ToTorchFormatTensor(), ])

        self.V2X_config_global = ConfigGlobal("train", binary=True, only_det=True)
        self.V2X_config = Config("train", binary=True, only_det=True)
        self.agent_data_dirs = []
        for agent_id in range(self.args.num_vehicles):
            self.agent_data_dirs.append(os.path.join(self.args.v2x_data_path, split, f"agent{agent_id+1}"))
        self.v2x_det_dataset = V2XSimDet(
            dataset_roots=self.agent_data_dirs, split=split, config_global=self.V2X_config_global,
            config=self.V2X_config , val=True, bound='both')
        LOG.info(f"v2x_det_dataset length: {len(self.v2x_det_dataset)}")
        LOG.info(f"v2x_det_dataset: {self.v2x_det_dataset}")

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in video {}'.format(self.video_names[index]))
            item = self.load_item(0)
        return item

    def load_item(self, index):
        video_name = self.video_names[index]
        all_frames = [f"{str(i).zfill(5)}.jpg" for i in range(self.video_dict[video_name])]
        all_masks = create_random_shape_with_random_motion(
            len(all_frames), imageHeight=self.h, imageWidth=self.w)
        ref_index = get_ref_index(len(all_frames), self.sample_length)
        # read video frames
        frames = []
        masks = []
        for idx in ref_index:
            img = ZipReader.imread('{}/{}/JPEGImages/{}.zip'.format(
                self.args['data_root'], self.args['name'], video_name), all_frames[idx]).convert('RGB')
            img = img.resize(self.size)
            frames.append(img)
            masks.append(all_masks[idx])
        if self.split == 'train':
            frames = GroupRandomHorizontalFlip()(frames)
        # To tensors
        frame_tensors = self._to_tensors(frames)*2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return frame_tensors, mask_tensors


