import gradio as gr
import numpy as np
import torch
import torchvision.transforms as transforms
import argparse

from transformers import DistilBertTokenizer
from model_pytorch import BertVistaNet
from photo_preprocess import make_vgg_model
from data_bert_vgg_loader import Bertvggdataset
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--dropout_keep_prob', type=float, default=0.5)
parser.add_argument('--num_images', type=int, default=3)
parser.add_argument('--num_classes', type=int, default=5)
parser.add_argument('--max_num_words', type=int, default=30)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=2e-5)
args = parser.parse_args()

checkpoint_path = 'checkpoints/epoch=1-loss=0.8303-acc=0.6470'

img_to_tensor = transforms.ToTensor()

####Model Load####
vision_model = make_vgg_model()
model = BertVistaNet(args.num_images, args.num_classes, args.dropout_keep_prob)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
checkpoint = torch.load(checkpoint_path)
# create new OrderedDict that does not contain `module.`
new_checkpoint = OrderedDict()
for k, v in checkpoint["model_state_dict"].items():
    name = k[7:] # remove `module.`
    new_checkpoint[name] = v
# load params
model.load_state_dict(new_checkpoint)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 
epoch = checkpoint["epoch"]

def features_extraction(imgs, vision_model):
    vision_model.eval()
    imgs = imgs.view(-1, 3, 224, 224)
    features = vision_model(imgs).view(-1, args.num_images, 4096)
    return features.detach()

def test(model, vision_model, input_ids, attention_mask, images):
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask, images)
        _, pred = torch.max(logits, dim=1)
        pred = pred.detach().numpy()
    return pred[0]

def loadall(text, img1, img2, img3):
    #####Text Load####
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encode = tokenizer.encode_plus(text, add_special_tokens=True, max_length=args.max_num_words, pad_to_max_length=False, truncation=True)
    input_ids = torch.tensor(encode['input_ids'])
    input_ids = input_ids.unsqueeze(0)
    attention_mask = torch.tensor(encode['attention_mask'])
    attention_mask = attention_mask.unsqueeze(0)
    
    #####Images Load####
    img1_tensor = img_to_tensor(img1)
    img2_tensor = img_to_tensor(img2)
    img3_tensor = img_to_tensor(img3)
    imgs = [img1_tensor, img2_tensor, img3_tensor]
    imgs = torch.stack(imgs)
    images = features_extraction(imgs, vision_model)
    
    pred = test(model, vision_model, input_ids, attention_mask, images)

    Semantic_Map = {
        0 : 'Extremely_Negative',
        1 : 'Negative',
        2 : 'Neutral',
        3 : 'Positive',
        4 : 'Extremely_Positive'
    }

    return "The semantic analysis result is " + str(Semantic_Map[pred]) + '.'

iface = gr.Interface(
    fn=loadall,
    inputs=[gr.inputs.Textbox(lines=5, placeholder="Write Some Comments Here..."), gr.inputs.Image(shape=(224, 224)), gr.inputs.Image(shape=(224, 224)), gr.inputs.Image(shape=(224, 224))],
    outputs=["text"])

iface.launch(share = True)