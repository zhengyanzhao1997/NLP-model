import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer,AlbertTokenizer
from mogai_bert_nezha import BertModel,BertConfig
from tqdm import tqdm
from torch.nn import functional as F
import os
import sys
from torch.nn import init
model_type = sys.argv[1]
assert model_type in ['nezha','wobert']

if model_type == 'nezha':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    model_path = "./nezha-base-www"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    Config = BertConfig.from_pretrained(model_path)
    Config.position_embedding_type = "nezha"
    Config.max_position_embeddings = 1024
    Config.conditional_size = 128
    maxlen = 1024
    batch_size = 16
    learning_rate = 1e-5
    num_folds = 5
    epochs = 3

elif model_type == 'wobert':
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    model_path = "./wobert-base"
    Config = BertConfig.from_pretrained(model_path)
    tokenizer = AlbertTokenizer.from_pretrained(model_path)
    Config.conditional_size = 128
    maxlen = 512
    batch_size = 16
    learning_rate = 1e-5
    num_folds = 5
    epochs = 5
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))

output_file = '%s_fold_test_result.csv' % model_type

variants = [
    u'短短匹配A类',
    u'短短匹配B类',
    u'短长匹配A类',
    u'短长匹配B类',
    u'长长匹配A类',
    u'长长匹配B类',
]

# 读取数据
train_data,test_data = [], []
for i, var in enumerate(variants):
    key = 'labelA' if 'A' in var else 'labelB'
    fs = [
        './datasets/sohu2021_open_data_clean/%s/train.txt' % var,
        './datasets/round2/%s.txt' % var,
        './datasets/round3/%s/train.txt' % var
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
            train_data.append((i, l['source'], l['target'], int(l[key])))

    f = './datasets/sohu2021_open_data_clean/%s/test_with_id.txt' % var
    with open(f) as f:
        for l in f:
            l = json.loads(l)
            test_data.append((i, l['source'], l['target'], l['id']))

print(len(train_data))

class CustomImageDataset(Dataset):
    
    def __init__(self, data, tokenizer, maxlen, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source, target, c):
        if c == 4 or c == 5:
            input_ids = np.zeros(self.maxlen, dtype='int')
            attention_mask = np.zeros(self.maxlen, dtype='int')
            token_type_ids = np.zeros(self.maxlen, dtype='int')
            one_maxlen = self.maxlen // 2
            token_id_1 = self.tokenizer.encode(source,max_length=one_maxlen,truncation=True)
            token_id_2 = self.tokenizer.encode(target,max_length=one_maxlen,truncation=True)
            input_id = token_id_1 + token_id_2[1:]
            token_type_id =  [0]*len(token_id_1) + [1]*(len(token_id_2)-1)
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
        return input_ids, attention_mask,token_type_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c = self.data[idx][0]
        text_source = self.data[idx][1]
        text_target = self.data[idx][2]
        label = self.data[idx][3]
        input_ids, attention_mask,token_type_ids = self.text_to_id(text_source, text_target, c)
        sample = {"input_ids": input_ids, "attention_mask": attention_mask,'token_type_ids':token_type_ids}
        return sample, label, c

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self,model_path):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
        self.standerembed = nn.Embedding(6,128)
        self.bert = BertModel.from_pretrained(model_path,config=Config)

    def forward(self, input_ids, attention_mask,token_type_ids,c):
        conditional = self.standerembed(c)
        x1 = self.bert(input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids,conditional=conditional)
        x2 = x1.last_hidden_state
        logits = self.linear_relu_stack(x2[:, 0])
        return logits

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    correct = 0
    for batch, (data,y,c) in enumerate(dataloader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        y = y.to(device)
        c = c.to(device)
        pred = model(input_ids,attention_mask,token_type_ids,c)
        loss = loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 200 == 0:
            loss, current = loss.item(), batch * len(input_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Accuracy: {(100*correct/size):>0.1f}%")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss = 0
    TP_a, TN_a, FN_a, FP_a = 0, 0, 0, 0
    TP_b, TN_b, FN_b, FP_b = 0, 0, 0, 0
    with torch.no_grad():
        for data,y,c in dataloader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            y = y.to(device)
            c = c.to(device)
            pred = model(input_ids,attention_mask,token_type_ids,c)
            test_loss += loss_fn(pred, y).item()
            pred_result = pred.argmax(1)
            TP_a += ((pred_result == 1) & (y == 1) & ((c==0) | (c==2) | (c==4))).type(torch.float).sum().item()
            TN_a += ((pred_result == 0) & (y == 0) & ((c==0) | (c==2) | (c==4))).type(torch.float).sum().item()
            FN_a += ((pred_result == 0) & (y == 1) & ((c==0) | (c==2) | (c==4))).type(torch.float).sum().item()
            FP_a += ((pred_result == 1) & (y == 0) & ((c==0) | (c==2) | (c==4))).type(torch.float).sum().item()
            TP_b += ((pred_result == 1) & (y == 1) & ((c==1) | (c==3) | (c==5))).type(torch.float).sum().item()
            TN_b += ((pred_result == 0) & (y == 0) & ((c==1) | (c==3) | (c==5))).type(torch.float).sum().item()
            FN_b += ((pred_result == 0) & (y == 1) & ((c==1) | (c==3) | (c==5))).type(torch.float).sum().item()
            FP_b += ((pred_result == 1) & (y == 0) & ((c==1) | (c==3) | (c==5))).type(torch.float).sum().item()
    test_loss /= size
    p_a = TP_a / (TP_a + FP_a)
    r_a = TP_a / (TP_a + FN_a)
    p_b = TP_b / (TP_b + FP_b)
    r_b = TP_b / (TP_b + FN_b)
    F1_a = 2 * r_a * p_a / (r_a + p_a)
    F1_b = 2 * r_b * p_b / (r_b + p_b)
    F1 = (F1_a+F1_b)/2
    print(f"Test Error: \n ,F1a_score:{(F1_a):>5f}, F1b_score:{(F1_b):>5f},\n F1_score:{(F1):>5f} ,Avg loss: {test_loss:>8f} \n")
    return F1

def data_split(data, fold, num_folds,mode):
    """划分训练集和验证集
    """
    if mode == 'train':
        D = [d for i, d in enumerate(data) if i % num_folds != fold]
    else:
        D = [d for i, d in enumerate(data) if i % num_folds == fold]
    return D

def init_model(model_base,mode,load_path=None):
    if mode == 'train':
        if model_base == 'wobert':
            model = NeuralNetwork(model_path).to(device)
            for i in model.state_dict():
                if 'LayerNorm.bias_dense' in i or 'LayerNorm.weight_dense' in i:
                    init.zeros_(model.state_dict()[i])

        elif model_base == 'nezha':
            model = NeuralNetwork(model_path)
            for i in model.state_dict():
                if 'LayerNorm.bias_dense' in i or 'LayerNorm.weight_dense' in i:
                    init.zeros_(model.state_dict()[i])
            model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()

    elif mode == 'pred' and load_path:
        model = NeuralNetwork(model_path).to(device)
        model.load_state_dict(torch.load(load_path))
    return model

def save_model(model_base,path):
    if model_base == 'wobert':
        torch.save(model.state_dict(), path)
    elif model_base == 'nezha':
        torch.save(model.module.state_dict(), path)

if __name__ == '__main__':

    np.random.shuffle(train_data)
    #training
    for fold in range(num_folds):
        torch.cuda.empty_cache()
        train_ = data_split(train_data,fold,num_folds,'train')
        valid_ = data_split(train_data,fold,num_folds,'valid')
        training_data = CustomImageDataset(train_, tokenizer, maxlen)
        validing_data = CustomImageDataset(valid_, tokenizer, maxlen)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(validing_data, batch_size=batch_size)
        model = init_model(model_type,'train')
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        F1max = 0
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, optimizer)
            test_F1 = test(valid_dataloader, model)
            if test_F1 > F1max:
                F1max = test_F1
                save_model(model_type,"./model_saved/Best_%s_model_%s.pth" % (model_type,fold))
                print(f"Higher F1: {(F1max):>5f}%, Saved PyTorch Model State to model.pth")
        print("Training Done!")

    #pred_test
    pred_result = {x[3]:0.0 for x in test_data}
    testing_data = CustomImageDataset(test_data, tokenizer, maxlen)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size)
    for fold in range(num_folds):
        torch.cuda.empty_cache()
        model = init_model(model_type,'pred',"./model_saved/Best_%s_model_%s.pth" % (model_type,fold))
        model.eval()
        print("predicting %s fold data" % fold)
        with torch.no_grad():
            for data, y_true in tqdm(test_dataloader):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                token_type_ids = data['token_type_ids'].to(device)
                pred = model(input_ids, attention_mask, token_type_ids)
                y_pred = F.softmax(pred,dim=-1)[:,1]
                for id, y in zip(y_true, y_pred):
                    pred_result[id] += y.item()/num_folds

    #write_out
    print("wrting out")
    with open(output_file, 'w') as f:
        f.write('id,label\n')
        for i in pred_result.keys():
            f.write('%s,%s\n' % (i,pred_result[i]))
