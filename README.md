# NLP-model
自己在学习与工作中搭建的NLP模型，论文复现或实际生产应用 

具体代码解读可以follow我的博客 https://blog.csdn.net/weixin_45839693


语言：Python 3.8

## 框架：Tensorflow 2.0 Transformers 3.1.0

目前更新的模型：

Sentence_bert NLP-model/model/TF_model/Train_Sentence-BERT.py

Bert-Last_3embedding_concat 情绪分类模型 NLP-model/model/TF_model/Train_Bert-Last_3embedding_concat_classification.py

SQuAD 2020语言与智能技术竞赛：机器阅读理解任务 baseline模型  NLP-model/model/TF_model/SQuAD_baseline.py

关系抽取——基于主语感知的层叠式指针网络 NLP-model/model/TF_model/Information_extraction/三元组抽取_指针标注.py

关系抽取——基于 Muti_head_selection NLP-model/model/TF_model/Information_extraction/关系抽取_Multi-head Selection.py

关系抽取——基于 Deep Biaffine Attention NLP-model/model/TF_model/Information_extraction/关系抽取_Deep Biaffine Attention.py 

Unified Language Model 新闻摘要生成 NLP-model/model/TF_model/Unified Language Model

NEZHA 相对位置模型（处理长文本）法律摘要生成 NLP-model/TF_model/model/NEZHA

SDP 2021@NAACL LongSumm 第一名 模型集合 NLP-model/model/TF_model/Longsumm



## 框架：torch 1.8.0 Transformers 4.1.5

目前更新的模型：

2021搜狐校园文本匹配算法大赛 P-tuning-Bert BaseLine NLP-model/model/Torch_model/Souhu_TextMatch

2021搜狐校园文本匹配算法大赛 Layer_conditional_norm BaseLine NLP-model/model/Torch_model/Souhu_TextMatch

SimCSE 论文复现 无监督/有监督对比学习 NLP-model/model/Torch_model/SimCSE-Chinese

嵌套实体命名识别 GlobalPointer、TPLinker、Tencent Muti-head、Deep Biaffine NLP-model/model/Torch_model/ExtractionEntities

