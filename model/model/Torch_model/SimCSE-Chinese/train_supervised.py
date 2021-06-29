import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
import scipy.stats
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))
model_path = "../model_set/chinese-bert-wwm-ext"
save_path = "./model_saved/128_pooler_supervised_best_model.pth"
tokenizer = BertTokenizer.from_pretrained(model_path)
Config = BertConfig.from_pretrained(model_path)

output_way = 'pooler'
batch_size = 128
learning_rate = 2e-5
maxlen = 64

sts_file_path = "./datasets/STS-B/"
sts_test_file = 'cnsd-sts-test.txt'
sts_dev_file = 'cnsd-sts-dev.txt'

snli_file_path = "./datasets/cnsd-snli/"
snli_train_file = 'cnsd_snli_v1.0.trainproceed.txt'

def load_snli_vocab(path):
    data = []
    with open(path) as f:
        for i in f:
            data.append(json.loads(i))
    return data

def load_STS_data(path):
    data = []
    with open(path) as f:
        for i in f:
            d = i.split("||")
            sentence1 = d[1]
            sentence2 = d[2]
            score = d[3]
            data.append([sentence1,sentence2,score])
    return data

snil_vocab = load_snli_vocab(os.path.join(snli_file_path,snli_train_file))
np.random.shuffle(snil_vocab)

test_data = load_STS_data(os.path.join(sts_file_path,sts_test_file))
dev_data = load_STS_data(os.path.join(sts_file_path,sts_dev_file))


class TrainDataset(Dataset):
    def __init__(self, data, tokenizer, maxlen, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source):
        origin = source['origin']
        entailment = source['entailment']
        contradiction = source['contradiction']
        sample = self.tokenizer([origin,entailment,contradiction],max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return sample

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.text_to_id(self.data[idx])


class TestDataset:
    def __init__(self, data, tokenizer, maxlen):
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.traget_idxs = self.text_to_id([x[0] for x in data])
        self.source_idxs = self.text_to_id([x[1] for x in data])
        self.label_list = [int(x[2]) for x in data]
        assert len(self.traget_idxs['input_ids']) == len(self.source_idxs['input_ids'])

    def text_to_id(self,source):
        sample = self.tokenizer(source,max_length=self.maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return sample

    def get_data(self):
        return self.traget_idxs,self.source_idxs,self.label_list


class NeuralNetwork(nn.Module):
    def __init__(self,model_path,output_way):
        super(NeuralNetwork, self).__init__()
        self.bert = BertModel.from_pretrained(model_path,config=Config)
        self.output_way = output_way
        assert output_way in ['cls','pooler']

    def forward(self, input_ids, attention_mask, token_type_ids):
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if self.output_way == 'cls':
            output = x1.last_hidden_state[:,0]
        elif self.output_way == 'pooler':
            output = x1.pooler_output
        return output

model = NeuralNetwork(model_path,output_way).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

training_data = TrainDataset(snil_vocab,tokenizer,maxlen)
train_dataloader = DataLoader(training_data,batch_size=batch_size)

testing_data = TestDataset(test_data,tokenizer,maxlen)
deving_data = TestDataset(dev_data,tokenizer,maxlen)

def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation

def compute_loss(y_pred,lamda=0.05):
    row = torch.arange(0,y_pred.shape[0],3,device='cuda')
    col = torch.arange(y_pred.shape[0], device='cuda')
    col = torch.where(col % 3 != 0)[0].cuda()
    y_true = torch.arange(0,len(col),2,device='cuda')
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    #torch自带的快速计算相似度矩阵的方法
    similarities = torch.index_select(similarities,0,row)
    similarities = torch.index_select(similarities,1,col)
    #屏蔽对角矩阵即自身相等的loss
    similarities = similarities / lamda
    #论文中除以 temperature 超参 0.05
    loss = F.cross_entropy(similarities,y_true)
    return torch.mean(loss)

def test(test_data,model):
    traget_idxs, source_idxs, label_list = test_data.get_data()
    with torch.no_grad():
        traget_input_ids = traget_idxs['input_ids'].to(device)
        traget_attention_mask = traget_idxs['attention_mask'].to(device)
        traget_token_type_ids = traget_idxs['token_type_ids'].to(device)
        traget_pred = model(traget_input_ids,traget_attention_mask,traget_token_type_ids)

        source_input_ids = source_idxs['input_ids'].to(device)
        source_attention_mask = source_idxs['attention_mask'].to(device)
        source_token_type_ids = source_idxs['token_type_ids'].to(device)
        source_pred = model(source_input_ids,source_attention_mask,source_token_type_ids)
       
        similarity_list = F.cosine_similarity(traget_pred,source_pred)
        similarity_list = similarity_list.cpu().numpy()
        label_list = np.array(label_list)
        corrcoef = compute_corrcoef(label_list,similarity_list)
    return corrcoef

def train(dataloader,testdata, model, optimizer):
    model.train()
    size = len(dataloader.dataset)
    max_corrcoef = 0
    not_up_batch = 0
    for batch, data in enumerate(dataloader):
        input_ids = data['input_ids'].view(len(data['input_ids'])*3,-1).to(device)
        attention_mask = data['attention_mask'].view(len(data['attention_mask'])*3,-1).to(device)
        token_type_ids = data['token_type_ids'].view(len(data['token_type_ids'])*3,-1).to(device)
        pred = model(input_ids,attention_mask,token_type_ids)
        loss = compute_loss(pred)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), batch * int(len(input_ids)/3)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            model.eval() #意味着显示关闭丢弃dropout
            corrcoef = test(testdata,model)
            model.train()
            print(f"corrcoef_test: {corrcoef:>4f}")
            if corrcoef > max_corrcoef:
                not_up_batch = 0
                max_corrcoef = corrcoef
                torch.save(model.state_dict(),save_path)
                print(f"Higher corrcoef: {(max_corrcoef):>4f}%, Saved PyTorch Model State to model.pth")
            else:
                not_up_batch += 1
                if not_up_batch > 10:
                    print(f"Corrcoef didn't up for %s batch, early stop!" % not_up_batch)
                    break

if __name__ == '__main__':
    epochs = 1
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader,testing_data, model, optimizer)
    print("Train_Done!")
    print("Deving_start!")
    model.load_state_dict(torch.load(save_path))
    corrcoef = test(deving_data,model)
    print(f"dev_corrcoef: {corrcoef:>4f}")
