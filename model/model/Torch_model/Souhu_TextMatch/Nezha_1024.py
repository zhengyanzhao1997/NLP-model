import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from mogai_bert_nezha import BertModel, BertConfig
import os
from torch.nn import init
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))
model_path = "./nezha-base-www"
#model_path = "./chinese-bert-wwm-ext"
tokenizer = BertTokenizer.from_pretrained(model_path)
Config = BertConfig.from_pretrained(model_path)
Config.conditional_size = 128
Config.position_embedding_type = "nezha"
Config.max_position_embeddings = 1024
batch_size = 16
maxlen = 1024
LR = 1e-5
variants = [
    u'短短匹配A类',
    u'短短匹配B类',
    u'短长匹配A类',
    u'短长匹配B类',
    u'长长匹配A类',
    u'长长匹配B类',
]

# 读取数据
train_data, valid_data, test_data = [], [], []
for i, var in enumerate(variants):
    key = 'labelA' if 'A' in var else 'labelB'
    fs = [
        './datasets/sohu2021_open_data_clean/%s/train.txt' % var,
        './datasets/round2/%s.txt' % var
    ]
    for f in fs:
        with open(f) as f:
            for l in f:
                l = json.loads(l)
                train_data.append((i, l['source'], l['target'], int(l[key])))
    f = './datasets/sohu2021_open_data_clean/%s/valid.txt' % var
    with open(f) as f:
        for l in f:
            l = json.loads(l)
            valid_data.append((i, l['source'], l['target'], int(l[key])))


class CustomImageDataset(Dataset):

    def __init__(self, data, tokenizer, maxlen, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.transform = transform
        self.target_transform = target_transform
        # self.c2stander = {0:0,1:1,2:0,3:1,4:0,5:1}
        # self.c2length = {0:0, 1:0, 2:1, 3: 1, 4: 2, 5: 2}

    def text_to_id(self, source, target, c):
        if c == 4 or c == 5:
            input_ids = np.zeros(self.maxlen, dtype='int')
            attention_mask = np.zeros(self.maxlen, dtype='int')
            token_type_ids = np.zeros(self.maxlen, dtype='int')
            one_maxlen = self.maxlen // 2
            token_id_1 = self.tokenizer.encode(source, max_length=one_maxlen, truncation=True)
            token_id_2 = self.tokenizer.encode(target, max_length=one_maxlen, truncation=True)
            input_id = token_id_1 + token_id_2[1:]
            token_type_id = [0] * len(token_id_1) + [1] * (len(token_id_2) - 1)
            assert len(input_id) == len(token_type_id)
            input_ids[:len(input_id)] = input_id
            attention_mask[:len(input_id)] = 1
            token_type_ids[:len(token_type_id)] = token_type_id
        else:
            input_ids = np.zeros(self.maxlen, dtype='int')
            attention_mask = np.zeros(self.maxlen, dtype='int')
            token_type_ids = np.zeros(self.maxlen, dtype='int')
            token_id = self.tokenizer(source, target, max_length=self.maxlen, truncation=True)
            token_type_id = token_id['token_type_ids']
            input_id = token_id['input_ids']
            assert len(input_id) == len(token_type_id)
            input_ids[:len(input_id)] = input_id
            attention_mask[:len(input_id)] = 1
            token_type_ids[:len(token_type_id)] = token_type_id
        return input_ids, attention_mask, token_type_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c = self.data[idx][0]
        text_source = self.data[idx][1]
        text_target = self.data[idx][2]
        label = self.data[idx][3]
        input_ids, attention_mask, token_type_ids = self.text_to_id(text_source, text_target, c)
        sample = {"input_ids": input_ids, "attention_mask": attention_mask, 'token_type_ids': token_type_ids}
        # stander = self.c2stander[c]
        # length = self.c2length[c]
        return sample, label, c


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, model_path):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.standerembed = nn.Embedding(6, 128)
        # self.lengthembed = nn.Embedding(3,32)
        self.bert = BertModel.from_pretrained(model_path, config=Config)

    def forward(self, input_ids, attention_mask, token_type_ids, c):
        conditional = self.standerembed(c)
        # lengthembed = self.lengthembed(length)
        # conditional = torch.cat([standerembed,lengthembed],-1)
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, conditional=conditional)
        x2 = x1.last_hidden_state
        logits = self.linear_relu_stack(x2[:, 0])
        return logits


model = NeuralNetwork(model_path)
print(model)
for i in model.state_dict():
    if 'LayerNorm.bias_dense' in i or 'LayerNorm.weight_dense' in i:
        init.zeros_(model.state_dict()[i])
model = nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()
#model = model.cuda()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

training_data = CustomImageDataset(train_data, tokenizer, maxlen)
testing_data = CustomImageDataset(valid_data, tokenizer, maxlen)
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    correct = 0
    model.train()
    for batch, (data, y, c) in enumerate(dataloader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        y = y.to(device)
        c = c.to(device)
        pred = model(input_ids, attention_mask, token_type_ids, c)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        print(pred.argmax(1))
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(input_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Accuracy: {(100 * correct / size):>0.1f}%")


def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    TP_a, TN_a, FN_a, FP_a = 0, 0, 0, 0
    TP_b, TN_b, FN_b, FP_b = 0, 0, 0, 0
    with torch.no_grad():
        for data, y, c in dataloader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            y = y.to(device)
            c = c.to(device)
            pred = model(input_ids, attention_mask, token_type_ids, c)
            test_loss += loss_fn(pred, y).item()
            pred_result = pred.argmax(1)
            TP_a += ((pred_result == 1) & (y == 1) & ((c == 0) | (c == 2) | (c == 4))).type(torch.float).sum().item()
            TN_a += ((pred_result == 0) & (y == 0) & ((c == 0) | (c == 2) | (c == 4))).type(torch.float).sum().item()
            FN_a += ((pred_result == 0) & (y == 1) & ((c == 0) | (c == 2) | (c == 4))).type(torch.float).sum().item()
            FP_a += ((pred_result == 1) & (y == 0) & ((c == 0) | (c == 2) | (c == 4))).type(torch.float).sum().item()
            TP_b += ((pred_result == 1) & (y == 1) & ((c == 1) | (c == 3) | (c == 5))).type(torch.float).sum().item()
            TN_b += ((pred_result == 0) & (y == 0) & ((c == 1) | (c == 3) | (c == 5))).type(torch.float).sum().item()
            FN_b += ((pred_result == 0) & (y == 1) & ((c == 1) | (c == 3) | (c == 5))).type(torch.float).sum().item()
            FP_b += ((pred_result == 1) & (y == 0) & ((c == 1) | (c == 3) | (c == 5))).type(torch.float).sum().item()
    test_loss /= size
    p_a = TP_a / (TP_a + FP_a)
    r_a = TP_a / (TP_a + FN_a)
    p_b = TP_b / (TP_b + FP_b)
    r_b = TP_b / (TP_b + FN_b)
    F1_a = 2 * r_a * p_a / (r_a + p_a)
    F1_b = 2 * r_b * p_b / (r_b + p_b)
    F1 = (F1_a + F1_b) / 2
    print(
        f"Test Error: \n ,F1a_score:{(F1_a):>5f}, F1b_score:{(F1_b):>5f},\n F1_score:{(F1):>5f} ,Avg loss: {test_loss:>8f} \n")
    return F1


if __name__ == '__main__':
    epochs = 5
    F1max = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test_F1 = test(test_dataloader, model)
        if test_F1 > F1max:
            F1max = test_F1
            torch.save(model.module.state_dict(), "./model_saved/nezha_Conditional_F1_%s_model.pth" % F1max)
            print(f"Higher F1: {(F1max):>5f}%, Saved PyTorch Model State to model.pth")
    print("Done!")
