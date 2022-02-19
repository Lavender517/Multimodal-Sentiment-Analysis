import numpy as np
import argparse
import torch
import torch.nn as nn
import os, pickle, json

from model_pytorch import BertVistaNet
from train_transformer import test
from data_preprocess import cities
from data_utils import batch_review_normalize, batch_image_normalize
from photo_preprocess import make_vgg_model
from data_bert_vgg_loader import Bertvggdataset
from torch.utils.data import DataLoader
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--dropout_keep_prob', type=float, default=0.5)
parser.add_argument('--num_images', type=int, default=3)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--max_num_words', type=int, default=400)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_ids = [0, 1]

checkpoint_path = 'checkpoints/epoch=1-loss=0.8303-acc=0.6470'

model = BertVistaNet(args.num_images, args.num_classes, args.dropout_keep_prob).to(device)
# model = torch.nn.DataParallel(model, device_ids)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

checkpoint = torch.load(checkpoint_path)
# create new OrderedDict that does not contain `module.`
new_checkpoint = OrderedDict()
for k, v in checkpoint["model_state_dict"].items():
    name = k[7:] # remove `module.`
    new_checkpoint[name] = v
# load params
model.load_state_dict(new_checkpoint)

# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 
epoch = checkpoint["epoch"]

criterion = nn.CrossEntropyLoss()
vision_model = make_vgg_model().to(device)

data_dir = 'data'
test_loader = {}
for city in cities:
    test_raw_path = '{}_test.json'.format(city)
    test_path = '{}_test.pickle'.format(city)
    test_raw_file = os.path.join(data_dir, 'test', test_raw_path)
    test_file = os.path.join(data_dir, 'test', test_path)
    test_data = Bertvggdataset(test_raw_file, test_file, args.num_images, args.max_num_words)
    test_loader[city] = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=16)

# model.eval()
def test_evaluate(criterion, model, cities, args, vision_model, test_loader):
    city_acc = []
    for city in cities:
        loss_test, acc_test, test_pred = test(criterion, model, city, args, vision_model, test_loader)
        city_acc.append(acc_test)
        print(f"city:{city}, test loss={loss_test}, test accuracy={acc_test}")
    test_acc_avg = (2080*city_acc[0] + 2165*city_acc[1] + 24860*city_acc[2] + 11425*city_acc[3] + 3775*city_acc[4]) / 44305
    print('Overall accuracy={:.4f}'.format(test_acc_avg))

test_evaluate(criterion, model, cities, args, vision_model, test_loader)