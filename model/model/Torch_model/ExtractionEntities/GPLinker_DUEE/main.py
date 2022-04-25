from transformers import BertTokenizerFast,BertModel
from torch import nn
from torch.utils.data import Dataset,DataLoader
import json
import os
import torch
from EfficientGlobalPointer import EfficientGlobalPointer as Globalmodel
from tqdm import tqdm
from itertools import groupby
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model_path = './pretrain_model'
tokenizer = BertTokenizerFast.from_pretrained(model_path)
maxlen = 128
batch_size = 32
epochs = 200
learning_rate = 2e-5


labels = []
with open('./data/duee_event_schema.json') as f:
    for l in f:
        l = json.loads(l)
        t = l['event_type']
        for r in [u'触发词'] + [s['role'] for s in l['role_list']]:
            labels.append((t, r))


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'events': [[(type, role, argument, start_index)]]}
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            d = {'text': l['text'], 'events': []}
            for e in l['event_list']:
                if e['trigger'].startswith(' '):
                    old_trigger = e['trigger']
                    e['trigger'] = e['trigger'].lstrip()
                    e['trigger_start_index'] += len(old_trigger) - len(e['trigger'])
                d['events'].append([(
                    e['event_type'], u'触发词', e['trigger'],
                    e['trigger_start_index']
                )])
                for a in e['arguments']:
                    if a['argument'].startswith(' '):
                        old_argument = a['argument']
                        a['argument'] = a['argument'].lstrip()
                        a['argument_start_index'] += len(old_argument) - len(a['argument'])
                    d['events'][-1].append((
                        e['event_type'], a['role'], a['argument'],
                        a['argument_start_index']
                    ))
            D.append(d)
    return D


# 加载数据集
train_data = load_data('./data/duee_train.json')
valid_data = load_data('./data/duee_dev.json')


class CustomDataset(Dataset):
    def __init__(self, data, labels, tokenizer, maxlen):
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    @staticmethod
    def find_index(offset_mapping, index):
        for idx, internal in enumerate(offset_mapping):
            if internal[0] <= index < internal[1]:
                return idx
        return None

    def get_label(self,d_events,offset_mapping):
        events = []
        for e in d_events:
            events.append([])
            for t, r, a, i in e:
                label = self.labels.index((t, r))
                start, end = i, i + len(a) - 1
                start_index = self.find_index(offset_mapping,start)
                end_index = self.find_index(offset_mapping,end)
                if start_index and end_index:
                    events[-1].append((label, start_index, end_index))
        argu_labels = torch.zeros((len(self.labels),self.maxlen,self.maxlen))
        head_labels = torch.zeros(self.maxlen,self.maxlen)
        tail_labels = torch.zeros(self.maxlen,self.maxlen)
        for e in events:
            for l, h, t in e:
                argu_labels[l, h, t] = 1
            for i1, (_, h1, t1) in enumerate(e):
                for i2, (_, h2, t2) in enumerate(e):
                    if i2 > i1:
                        head_labels[min(h1, h2), max(h1, h2)] = 1
                        tail_labels[min(t1, t2), max(t1, t2)] = 1
        return argu_labels,head_labels,tail_labels


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        token = tokenizer(d['text'],return_offsets_mapping=True,max_length=maxlen,truncation=True,padding='max_length',return_tensors='pt')
        input_ids = token['input_ids'][0]
        attention_mask = token['attention_mask'][0]
        offset_mapping = token['offset_mapping'][0]
        argu_labels,head_labels,tail_labels = self.get_label(d['events'],offset_mapping)
        return input_ids,attention_mask,argu_labels,head_labels,tail_labels


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
    #y_pred = (batch,c,l,l)
    b,c,m,n = y_pred.shape
    bh = m*n
    y_true = torch.reshape(y_true, (b, -1, m*n))
    y_pred = torch.reshape(y_pred, (b, -1, m*n))
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return torch.mean(torch.sum(loss,dim=1))


class ExtractModel(nn.Module):
    def __init__(self,model_path,argu_num,head_size):
        super(ExtractModel, self).__init__()
        self.argu_head = Globalmodel(heads=argu_num, head_size=head_size, hidden_size=768)
        self.head_head = Globalmodel(heads=1, head_size=head_size, hidden_size=768, RoPE=False)
        self.tail_head = Globalmodel(heads=1, head_size=head_size, hidden_size=768, RoPE=False)
        self.roberta = BertModel.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids, attention_mask=attention_mask)
        hidden_state = output.last_hidden_state
        argu_pred = self.argu_head(hidden_state,mask=attention_mask)
        head_pred = self.head_head(hidden_state,mask=attention_mask)
        tail_pred = self.tail_head(hidden_state,mask=attention_mask)
        return argu_pred,head_pred,tail_pred


class DedupList(list):
    """定义去重的list
    """
    def append(self, x):
        if x not in self:
            super(DedupList, self).append(x)


def neighbors(host, argus, links):
    """构建邻集（host节点与其所有邻居的集合）
    """
    results = [host]
    for argu in argus:
        # 遍历所有其他节点找出与host相邻的节点
        if host[2:] + argu[2:] in links:
            results.append(argu)
    return list(sorted(results))
    #返回host以及其所有相邻节点的集合


def clique_search(argus, links):
    # links 所有存在的边
    """搜索每个节点所属的完全子图作为独立事件
    搜索思路：找出不相邻的节点，然后分别构建它们的邻集，递归处理。
    递归思路从大节点往下找，找到不相邻的节点，按每个节点继续往下，直到
    对于一个节点其相邻节点之间都相邻，即完全子图，返回
    """
    Argus = DedupList()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if (h1, t1, h2, t2) not in links:
                    # i1 与 i2 为 不相邻节点 需要分别构建他们的邻集 去重
                    Argus.append(neighbors(argus[i1], argus, links))
                    Argus.append(neighbors(argus[i2], argus, links))
    if Argus:
        # 如果存在不相邻节点
        results = DedupList()
        for A in Argus:
            # 按新的不相邻节点以及其对应的相邻节点集合继续向下递归
            for a in clique_search(A, links):
                # 当其中一个节点为完全子图时会返回跳出
                results.append(a)
        return results
    else:
        # 则直接返回完全子图到上一层
        return [list(sorted(argus))]


def index2word(offset_mapping, start_index,end_index):
    start,_ = offset_mapping[start_index]
    _,end = offset_mapping[end_index]
    return start,end


def extract_events(text,model,threshold=0,trigger=True):
    """抽取输入text所包含的所有事件
    """
    tokens = tokenizer(text,return_offsets_mapping=True,max_length=maxlen,truncation=True,return_tensors='pt')
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    offset_mapping = tokens['offset_mapping'][0]
    outputs = model(input_ids,attention_mask)
    outputs = [o[0] for o in outputs]
    # 抽取论元
    argus = set()
    outputs[0][:,[0,-1]] -= 1e12
    outputs[0][:,:,[0,-1]] -= 1e12
    for l, h, t in zip(*torch.where(outputs[0] > threshold)):
        argus.add(labels[l] + (h, t))
        # (事件类型， 论元类型/触发词 ， 头index，尾index)
    # 构建链接
    links = set()
    for i1, (_, _, h1, t1) in enumerate(argus):
        for i2, (_, _, h2, t2) in enumerate(argus):
            if i2 > i1:
                if outputs[1][0, min(h1, h2), max(h1, h2)] > threshold:
                    if outputs[2][0, min(t1, t2), max(t1, t2)] > threshold:
                        links.add((h1, t1, h2, t2))
                        links.add((h2, t2, h1, t1))
    # 析出事件
    events = []
    for _, sub_argus in groupby(sorted(argus), key=lambda s: s[0]):
        #以同一事件类型进行最小完全子图搜索（不同事件类型的论元一定不是同一个事件）
        for event in clique_search(list(sub_argus), links):
            events.append([])
            for argu in event:
                start, end = index2word(offset_mapping,argu[2],argu[3])
                events[-1].append(argu[:2] + (text[start:end], start))
            if trigger and all([argu[1] != u'触发词' for argu in event]):
                events.pop()
    return events


def evaluate(data,model,threshold=0):
    """评估函数，计算f1、precision、recall
    """
    ex, ey, ez = 1e-10, 1e-10, 1e-10  # 事件级别
    ax, ay, az = 1e-10, 1e-10, 1e-10  # 论元级别
    for d in tqdm(data, ncols=0):
        pred_events = extract_events(d['text'], model, threshold, False)
        # 事件级别
        R, T = DedupList(), DedupList()
        for event in pred_events:
            if any([argu[1] == u'触发词' for argu in event]):
                R.append(list(sorted(event)))
        for event in d['events']:
            T.append(list(sorted(event)))
        for event in R:
            if event in T:
                ex += 1
        ey += len(R)
        ez += len(T)
        # 论元级别
        R, T = DedupList(), DedupList()
        for event in pred_events:
            for argu in event:
                if argu[1] != u'触发词':
                    R.append(argu)
        for event in d['events']:
            for argu in event:
                if argu[1] != u'触发词':
                    T.append(argu)
        for argu in R:
            if argu in T:
                ax += 1
        ay += len(R)
        az += len(T)
    e_f1, e_pr, e_rc = 2 * ex / (ey + ez), ex / ey, ex / ez
    a_f1, a_pr, a_rc = 2 * ax / (ay + az), ax / ay, ax / az
    return e_f1, e_pr, e_rc, a_f1, a_pr, a_rc



def global_pointer_f1_score(y_true, y_pred):
    y_pred = torch.greater(y_pred, 0)
    return torch.sum(y_true * y_pred).item(), torch.sum(y_true + y_pred).item()



def train(dataloader, model, optimizer):
    model.train()
    size = len(dataloader.dataset)
    numerate, denominator = 0, 0
    for batch, (input_ids,attention_mask,argu_labels,head_labels,tail_labels) in tqdm(enumerate(dataloader),total=(size//batch_size)):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        argu_labels = argu_labels.to(device)
        head_labels = head_labels.to(device)
        tail_labels = tail_labels.to(device)
        pred = model(input_ids,attention_mask)
        argu_loss = global_pointer_crossentropy(argu_labels, pred[0])
        head_loss = global_pointer_crossentropy(head_labels, pred[1])
        tail_loss = global_pointer_crossentropy(tail_labels, pred[2])
        loss = argu_loss + head_loss + tail_loss
        temp_n,temp_d = global_pointer_f1_score(argu_labels, pred[0])
        numerate += temp_n
        denominator += temp_d
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(input_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print(f"loss: {loss:>7f} argu loss:{argu_loss:>7f} head loss:{head_loss:>7f} tail loss:{tail_loss:>7f}")
    print(f"Train argu F1: {(2*numerate/denominator):>4f}")


@torch.no_grad()
def test(test_data, model):
    model.eval()
    e_f1, e_pr, e_rc, a_f1, a_pr, a_rc = evaluate(test_data,model)
    return e_f1, e_pr, e_rc, a_f1, a_pr, a_rc


if __name__ == '__main__':
    model = ExtractModel(model_path,len(labels),64).to(device)
    model.load_state_dict(torch.load('./best_model_3'))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_data = CustomDataset(train_data, labels, tokenizer, maxlen)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    max_F1 = 0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, optimizer)
        e_f1, e_pr, e_rc, a_f1, a_pr, a_rc = test(valid_data, model)
        print(e_f1, e_pr, e_rc)
        print(a_f1, a_pr, a_rc)
        if e_f1 > max_F1:
            max_F1 = e_f1
            print(f"Higher F1: {(max_F1):>4f}")
            torch.save(model.state_dict(),'./best_model_3')
    print("Done!")