import jsonlines
import os
import json
file_path = "./datasets/cnsd-snli/"
train_file = 'cnsd_snli_v1.0.train.jsonl'
test_file = 'cnsd_snli_v1.0.test.jsonl'
dev_file = 'cnsd_snli_v1.0.dev.jsonl'

def data_porcess(path):
    data_entailment = []
    data_contradiction = []
    with open(path, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            if item['gold_label'] == 'entailment':
                data_entailment.append(item)
            elif item['gold_label'] == 'contradiction':
                data_contradiction.append(item)
    data_entailment = sorted(data_entailment,key= lambda x:x['sentence1'])
    data_contradiction = sorted(data_contradiction, key=lambda x: x['sentence1'])
    process = []
    i = 0
    j = 0
    while i < len(data_entailment):
        origin = data_entailment[i]['sentence1']
        for index in range(j,len(data_contradiction)):
            if  data_entailment[i]['sentence1'] == data_contradiction[index]['sentence1']:
                process.append({'origin':origin,'entailment':data_entailment[i]['sentence2'],'contradiction':data_contradiction[index]['sentence2']})
                j = index + 1
                break
        while i < len(data_entailment) and data_entailment[i]['sentence1'] == origin:
            i += 1
        print(i)
    with open(path[:-6]+'proceed.txt','w') as f:
        for d in process:
            f.write(json.dumps(d) + '\n')

data_porcess(os.path.join(file_path,train_file))
data_porcess(os.path.join(file_path,test_file))
data_porcess(os.path.join(file_path,dev_file))


