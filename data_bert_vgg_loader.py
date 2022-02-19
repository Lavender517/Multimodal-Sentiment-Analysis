import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from data_preprocess import data_dir, train_file, valid_file, cities
from data_utils import batch_review_normalize, batch_image_normalize
from photo_preprocess import batch_image_load, load_img
from transformers import BertTokenizer
from transformers import DistilBertTokenizer

import numpy as np
import os, pickle, json
from tqdm import tqdm

NUM_SENTENCES = 30
NUM_WORDS = 30
num_images = 3
photo_dir = 'photos'

class Bertvggdataset(Dataset):
    def __init__(self, text_path, file_path, num_images=3, max_length=30):
        super().__init__()
        self.text_path=text_path
        self.file_path=file_path
        self.num_images=num_images
        self.max_length = max_length
        model_name = 'bert-base-uncased'
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.data = self._read_data(text_path, file_path)
    
    def _read_data(self, text_path, file_path):
        print('Reading data from %s and %s' %(text_path, file_path))
        data = []
        reviews = []
        # all_reviews = []
        with open(text_path, 'r') as t:
            print("Reading raw texts")
            for line in t:
                raw_review = json.loads(line)
                raw_review = raw_review['Text']
                raw_review = raw_review.split('|||')
                review_cat = ''
                for i in raw_review:
                    # review_cat = review_cat + ' [SEP] ' + i
                    review_cat = review_cat  + i
                # review_cat = review_cat[7:]
                reviews.append(review_cat)
        with open(file_path, 'rb') as f:
            try:
                while True:
                    review, images, rating = pickle.load(f)

                    # # clip review to specified max lengths
                    # review = review[:NUM_SENTENCES]
                    # review = [sent[:NUM_WORDS] for sent in review]
                    # all_reviews.append(review)

                    images = images[:self.num_images]
                    # images = batch_image_normalize([images], self.num_images)[0]

                    rating -= 1
                    assert rating >= 0 and rating <= 4

                    data.append([review, images, rating])
                
            except EOFError:
                # reviews, _, _, _, _ = batch_review_normalize(reviews)
                # print(all_reviews)
                for i, d in enumerate(data):
                    data[i][0] = reviews[i]
                return data
    
    def __getitem__(self, idx):
        imgs = []
        for photo_id in self.data[idx][1]:
            photo_path = os.path.join(photo_dir, photo_id[:2], photo_id) + '.jpg'
            imgs.append(load_img(photo_path))
        imgs = torch.stack(imgs)
        encode = self.tokenizer.encode_plus(self.data[idx][0], add_special_tokens=True, max_length=self.max_length, pad_to_max_length=True, truncation=True)
        return torch.tensor(encode['input_ids']), torch.tensor(encode['attention_mask']), imgs, self.data[idx][2]

    def __len__(self):
        return len(self.data)