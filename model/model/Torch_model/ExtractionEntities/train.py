import re
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer, BertModel, BertConfig,PreTrainedTokenizerFast
from head import GlobalPointer,MutiHeadSelection,Biaffine,TxMutihead
import sys
import os

head_type = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))
model_path = "../model_set/bert-base-chinese"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

assert head_type in ['GlobalPointer','MutiHeadSelection','Biaffine','TxMutihead']

if head_type in ['MutiHeadSelection','Biaffine','TxMutihead']:
    batch_size = 4
    learning_rate = 1e-5
    abPosition = False
    rePosition = True
else:
    batch_size = 16
    learning_rate = 2e-5

maxlen=256

def load_data(filename):
    data = []
    cat = set()
    with open(filename) as f:
        for text in f:
            text_list = text.replace('\n','').split("|||")[:-1]
            assert len(text_list) >= 2
            context = text_list[0]
            data.append([context])
            for e in text_list[1:]:
                e = re.sub('\s+', ' ', e)
                info = e.split(" ")
                assert len(info) == 3
                data[-1].append([int(info[0]),int(info[1]),context[int(info[0]):int(info[1])+1],info[2]])
                cat.add(info[2])
    return data,cat

train_data, cat = load_data('./datasets/train_data.txt')
val_data, _ = load_data('./datasets/val_data.txt')

c_size = len(cat)
c2id = {c:idx for idx,c in enumerate(cat)}
id2c = {idx:c for idx,c in enumerate(cat)}

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    @staticmethod
    def find_index(offset_mapping, index):
        for idx, internal in enumerate(offset_mapping[1:]):
            if internal[0] <= index < internal[1]:
                return idx + 1
        return None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        label = torch.zeros((c_size,self.maxlen,self.maxlen))
        enc_context = tokenizer(d[0],return_offsets_mapping=True,max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        enc_context = {key:enc_context[key][0] for key in enc_context.keys() if enc_context[key].shape[0] == 1}
        for entity_info in d[1:]:
            start, end = entity_info[0], entity_info[1]
            offset_mapping = enc_context['offset_mapping']
            start = self.find_index(offset_mapping, start)
            end = self.find_index(offset_mapping, end)
            if start and end and start < self.maxlen and end < self.maxlen:
                label[c2id[entity_info[3]],start,end] = 1
        return enc_context,label


class Net(nn.Module):
    def __init__(self,model_path,head_type):
        super(Net, self).__init__()
        if head_type == 'GlobalPointer':
            self.head = GlobalPointer(c_size, 64, 768)
        elif head_type == 'MutiHeadSelection':
            self.head = MutiHeadSelection(768,c_size,abPosition=abPosition,rePosition=rePosition,maxlen=maxlen,max_relative=64)
        elif head_type == 'Biaffine':
            self.head = Biaffine(768, c_size,Position=abPosition)
        elif head_type == 'TxMutihead':
            self.head = TxMutihead(768, c_size,abPosition=abPosition,rePosition=rePosition,maxlen=maxlen,max_relative=64)
        self.bert = BertModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x2 = x1.last_hidden_state
        logits = self.head(x2, mask = attention_mask)
        return logits

model = Net(model_path,head_type).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
training_data = CustomDataset(train_data,tokenizer,maxlen)
testing_data = CustomDataset(val_data,tokenizer,maxlen)
train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)


def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


def global_pointer_crossentropy(y_true, y_pred):
    """给GlobalPointer设计的交叉熵
    """
    #y_pred = (batch,l,l,c)
    bh = y_pred.shape[0] * y_pred.shape[1]
    y_true = torch.reshape(y_true, (bh, -1))
    y_pred = torch.reshape(y_pred, (bh, -1))
    return torch.mean(multilabel_categorical_crossentropy(y_true, y_pred))


def global_pointer_f1_score(y_true, y_pred):
    y_pred = torch.greater(y_pred, 0)
    return torch.sum(y_true * y_pred).item(), torch.sum(y_true + y_pred).item()


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    numerate, denominator = 0, 0
    for batch, (data,y) in enumerate(dataloader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        y = y.to(device)
        pred = model(input_ids,attention_mask,token_type_ids)
        loss = loss_fn(y,pred)
        temp_n,temp_d = global_pointer_f1_score(y,pred)
        numerate += temp_n
        denominator += temp_d
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(input_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Train F1: {(2*numerate/denominator):>4f}%")

def test(dataloader,loss_fn, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    numerate, denominator = 0, 0
    with torch.no_grad():
        for data,y in dataloader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            y = y.to(device)
            pred = model(input_ids, attention_mask, token_type_ids)
            test_loss += loss_fn(y,pred).item()
            temp_n, temp_d = global_pointer_f1_score(y, pred)
            numerate += temp_n
            denominator += temp_d
    test_loss /= size
    test_f1 = 2*numerate/denominator
    print(f"Test Error: \n ,F1:{(test_f1):>4f},Avg loss: {test_loss:>8f} \n")
    return test_f1

if __name__ == '__main__':
    epochs = 10
    max_F1 = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, global_pointer_crossentropy, optimizer)
        F1 = test(test_dataloader,global_pointer_crossentropy, model)
        if F1 > max_F1:
            max_F1 = F1
            print(f"Higher F1: {(max_F1):>4f}%")
    print("Done!")
