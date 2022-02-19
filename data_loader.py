import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from data_preprocess import data_dir, train_file, valid_file, cities
from data_utils import batch_review_normalize, batch_image_normalize

import numpy as np
import os, pickle

NUM_SENTENCES = 30
NUM_WORDS = 30
num_images = 3

class Mydataset(Dataset):
    def __init__(self, file_path, num_images=3):
        super().__init__()
        self.file_path=file_path
        self.num_images=num_images
        self.data = self._read_data(file_path)
        # print(self.data[0][0])
    
    def _read_data(self, file_path):
        print('Reading data from %s' % file_path)
        data = []
        all_reviews = []
        with open(file_path, 'rb') as f:
            try:
                while True:
                    review, images, rating = pickle.load(f)

                    # clip review to specified max lengths
                    review = review[:NUM_SENTENCES]
                    review = [sent[:NUM_WORDS] for sent in review]
                    all_reviews.append(review)
                    # review, _, _, _, _ = batch_review_normalize([review])

                    images = images[:self.num_images]
                    images = batch_image_normalize([images], self.num_images)[0]

                    rating -= 1
                    assert rating >= 0 and rating <= 4

                    data.append([review, images, rating])
                
            except EOFError:
                all_reviews, _, _, _, _ = batch_review_normalize(all_reviews)
                # print(all_reviews)
                for i, d in enumerate(data):
                    data[i][0] = all_reviews[i]
                return data
    
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.data[idx][2]

    def __len__(self):
        return len(self.data)
