import os

import torch
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class TwoStreamDataset(Dataset):

    def __init__(self, df, data_dir_1, data_dir_2):
        super(TwoStreamDataset, self).__init__()

        self.data_dir_1 = data_dir_1
        self.data_dir_2 = data_dir_2
        self.df = df


    def __len__(self):
        return len(self.df.values)

    def __getitem__(self, idx):
        label = self.df['label'].values[idx]

        slide_id = self.df['slide_id'].values[idx]

        features_path_1 = os.path.join(self.data_dir_1, 'patch_feature', slide_id + '.pt')
        features_path_2 = os.path.join(self.data_dir_2, 'patch_feature', slide_id + '.pt')
        features_1 = torch.load(features_path_1, map_location=torch.device('cpu'))
        features_2 = torch.load(features_path_2, map_location=torch.device('cpu'))

        coords_path_1 = os.path.join(self.data_dir_1, 'patch_coord', slide_id + '.h5')
        coords_path_2 = os.path.join(self.data_dir_2, 'patch_coord', slide_id + '.h5')
        coords_1 = h5py.File(coords_path_1, 'r')['coords'][:]
        coords_2 = h5py.File(coords_path_2, 'r')['coords'][:]

        return {
            'features_1': features_1,
            'features_2': features_2,
            'coords_1': coords_1,
            'coords_2': coords_2,
            'label': torch.tensor([label])

        }
    def get_data_df(self):
        return self.df
