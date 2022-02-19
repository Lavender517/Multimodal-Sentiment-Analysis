import torch
from data_utils import batch_review_normalize, batch_image_normalize
from model_utils_pytorch import load_glove
from data_preprocess import VOCAB_SIZE

import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from transformers import DistilBertModel
class BertVistaNet(nn.Module):
    def __init__(self, num_images, num_classes, dropout_keep_prob):
        super(BertVistaNet, self).__init__()
        self.num_classes = num_classes
        self.num_images = num_images
        self.dropout_keep_prob = dropout_keep_prob
        self.dropout = nn.Dropout(dropout_keep_prob)
        self.config = BertConfig()
        # self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.vis_attention = Visual_Aspect_Attention(self.config.hidden_size)
        # self.classifier = nn.Linear(self.config.hidden_size, self.num_classes)
        
        self.visual_transform = nn.Linear(4096, self.config.hidden_size)
        self.classifier = nn.Linear(2 * self.config.hidden_size, self.num_classes)
    def forward(self, input_ids, attention_mask, images):
        
        # text_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[1]
        text_emb = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
        # print(text_emb.shape)
        # sent_with_vis = self.vis_attention(text_emb, images)
        # doc_emb = self.doc_attention(sent_with_vis)
        images = torch.mean(images, dim=1, keepdim=False)
        images = torch.tanh(self.visual_transform(images))
        features = torch.cat((text_emb, images), 1)
        # print("features shape is", features.shape)
        pred = self.classifier(features)  
        # pred = self.classifier(text_emb)
        return pred

class VistaNet(nn.Module):
    def __init__(self, hidden_dim, att_dim, emb_size, num_images, num_classes, max_num_words, dropout_keep_prob):
        super(VistaNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.att_dim = att_dim
        self.emb_size = emb_size
        self.num_classes = num_classes
        self.num_images = num_images
        self.max_num_words = max_num_words
        self.dropout_keep_prob = dropout_keep_prob
        self.dropout = nn.Dropout(dropout_keep_prob)

        self._init_embedding()
        self._init_word_encoder()
        self._init_soft_attention()
        self._init_sent_encoder()
        self._init_vis_attention()
        self._init_doc_attention()
        self._init_classifier()
    
    def _init_embedding(self):
        init_weight = load_glove(VOCAB_SIZE, self.emb_size)
        init_weight = torch.FloatTensor(init_weight)
        self.embedding_matrix = nn.Embedding(VOCAB_SIZE, self.emb_size, _weight=init_weight)
        # print(self.embedding_matrix.weight.shape)

    def _init_word_encoder(self):
        self.word_encoder = nn.LSTM(input_size=self.emb_size, hidden_size=self.hidden_dim, bidirectional=True, batch_first=True, dropout=self.dropout_keep_prob)

    def _init_soft_attention(self):
        self.soft_attention = Soft_Attention(self.hidden_dim)

    def _init_sent_encoder(self):
        self.sent_encoder = nn.LSTM(input_size=self.hidden_dim * 2, hidden_size=self.hidden_dim, bidirectional=True, batch_first=True, dropout=self.dropout_keep_prob)
    
    def _init_vis_attention(self):
        self.vis_attention = Visual_Aspect_Attention(self.hidden_dim)

    def _init_doc_attention(self):
        self.doc_attention = Soft_Attention(self.hidden_dim)

    def _init_classifier(self):
        self.classifier = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, norm_docs, doc_sizes, max_num_sents, max_num_words, images, labels, hidden_word, hidden_sent):
        word_emb = self.embedding_matrix(norm_docs)
        word_emb = word_emb.view(-1, max_num_words, self.emb_size)
        # word_encode, _ = self.word_encoder(word_emb, hidden_word)
        word_encode, _ = self.word_encoder(word_emb)
        word_encode = self.dropout(word_encode)
        # print(word_encode.shape)
        att_word_encode = self.soft_attention(word_encode)
        # print(att_word_encode.shape)
        sent_emb = att_word_encode.view(-1, max_num_sents, self.hidden_dim * 2)
        # sent_encode, _ = self.sent_encoder(sent_emb, hidden_sent)
        sent_encode, _ = self.sent_encoder(sent_emb)
        sent_encode = self.dropout(sent_encode)
        sent_with_vis = self.vis_attention(sent_encode, images)
        doc_emb = self.doc_attention(sent_with_vis)
        pred = self.classifier(doc_emb)
        return pred

class Soft_Attention(nn.Module):
    def __init__(self, num_hiddens):
        super(Soft_Attention, self).__init__()
        # self.w_omega = nn.Parameter(torch.Tensor(num_hiddens * 2, num_hiddens * 2))
        self.w_omega = nn.Linear(num_hiddens * 2, num_hiddens * 2)
        # self.u_omega = nn.Parameter(torch.Tensor(num_hiddens * 2, 1))
        self.u_omega = nn.Linear(num_hiddens * 2, 1, bias=False)

        # nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, inputs):
        # u = torch.tanh(torch.matmul(inputs, self.w_omega))
        u = torch.tanh(self.w_omega(inputs))
        # att = torch.matmul(u, self.u_omega)
        att = self.u_omega(u)
        att_score = F.softmax(att, dim=1)
        scored_x = inputs * att_score
        feat = torch.sum(scored_x, dim=1)

        return feat

class Visual_Aspect_Attention(nn.Module):
    def __init__(self, num_hiddens):
        super(Visual_Aspect_Attention, self).__init__()
        self.num_hiddens = num_hiddens
        self.w_q = nn.Linear(num_hiddens * 2, num_hiddens * 2)
        self.w_p = nn.Linear(4096, num_hiddens * 2)
        # self.u_omega = nn.Parameter(torch.Tensor(num_hiddens * 2, 1))
        self.u_omega = nn.Linear(num_hiddens * 2, 1, bias=False)

        # nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, input_text, input_img):
        u_text = torch.tanh(self.w_q(input_text))
        n_t = u_text.shape[1]
        u_img = torch.tanh(self.w_p(input_img))
        n_v = u_img.shape[1]
        u_text = u_text.view(-1, 1, n_t, self.num_hiddens * 2)
        u_img = u_img.view(-1, n_v, 1, self.num_hiddens * 2)
        # print(u_img.shape)
        # print(u_text.shape)
        context = torch.mul(u_text, u_img) + u_text
        # print(context.shape)

        # att = torch.matmul(context, self.u_omega)
        att = self.u_omega(context)
        att_score = F.softmax(att, dim=1)
        input_text = input_text.view(-1, 1, n_t, self.num_hiddens * 2)
        # print(att_score.shape)
        # print(input_text.shape)
        scored_x = input_text * att_score
        feat = torch.sum(scored_x, dim=2)

        return feat