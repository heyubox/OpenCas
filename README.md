# An Information Cascade Project

This repository is an implementation of our novel model and the collection of information cascade model we compare wtih

## Features

* **Configurable**. The origianl model are fixed and the assumption of each model differs. In the `config.py`, you can config the model hyperparameters and easily preprocess the raw data.
* **Comparable**. Running the model with the same preprocess and model hyperparameters you can easily compare each model fairly.
* **Datasets**. We are going to run the models on more datasets. Now with `dataset_weibo.txt`, the model may generate many preprocessing file, we reorganize the floder.
* **Environment**. The required environment is in `requirements.txt`

## OpenCas Models

### How to run?

#### File Structure

```
.
├── README.md
├── models
│   ├── CasCN
│   ├── CasFlow
│   ├── DeepCas
│   ├── DeepHawkes
│   ├── TopoLSTM
│   ├── BSACas
│   └── model_save
└── rawdata
    └── dataset_weibo.txt
```

<!-- 1. 修改每一个model文件夹下的config.py文件，确保rawdata_root（原始数据的文件夹）,rawdataset（原始数据文件名，最后文件会从rawdata_root/rawdataset中读取）,data_root（中间文件生成的文件夹根目录）,dataset（中间文件生成的文件夹根目录下的子目录，针对不同任务有不同的文件夹）配置正确。**建议rawdata_root使用../../rawdata/，即放在和models文件夹同等级的目录下** -->
1. Pay attention to the file path. 

```python
ge_op.add_option("--rawdata_root", dest="rawdata_root", type="string", default="../../rawdata/", help="raw dataset root")
ge_op.add_option("--rawdataset", dest="rawdataset", type="string", default="dataset_weibo.txt", help="raw data set")
ge_op.add_option("--data_root", dest="data_root", type="string", default="/data/deephawkes/", help="data root.")
ge_op.add_option("--dataset", dest="dataset", type="string", default="citation1/", help="data set.")
```


1. A sample to run the  model:
```bash
cd models/BSACas
python -u gen_cas.py --rawdataset=dataset_weibo.txt --dataset=weibo/ --observation_time=7200 --interval=180 --up_num=100
python -u run.py --rawdataset=dataset_weibo.txt --dataset=weibo/ --observation_time=7200 --interval=180 --up_num=100
```
or
```bash
sh models/targetModel/run.sh
```

### BASCas

#### requirement

```
torch==^1.9.0
sklearn==^1.0.1
torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv at https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```

#### parameters
```
learning rate :0.001
num_mlp_layers : 1
num_layers : 2
gnn input_features : 100
gnn output_features : 64
input of rnn/transformer : 64
hidden_size (out put of rnn/tranformer) : 64
observation hour [7,19]
observation interval : 90
observation threshold : 1800
prediction time : 86400
cascade length [10,100]
```
#### quick run(default weibos)
```bash
cd models/cas2vec
python -u gen_cas.py 
python -u run.py
```

### [DeepHawkes](https://github.com/CaoQi92/DeepHawkes)

An implementation of DeepHawkes model in the following paper:

```text
Qi Cao, Huawei Shen, Keting Cen, Wentao Ouyang, Xueqi Cheng. 2017. DeepHawkes: Bridging the Gap between 
Prediction and Understanding of Information Cascades. In Proceedings of CIKM'17, Singapore., November 
6-10, 2017, 11 pages.
```

#### requirement
```
tensorflow-gpu==1.14.0
python==3.6
```
#### parameters
```
dropout prob :  0.001
l2 0.05
learning rate :  0.005
emb_learning_rate :  0.0005
observation hour [7,19]
observation threshold :  1800
prediction time :  86400
cascade length [10,100]
```

#### quick run(default weibos)
```bash
cd models/DeepHawkes
python -u gen_cas.py
python -u gen_run.py
python -u run.py
```

### [CasFlow](https://github.com/Xovee/casflow)

An reference implementation of CasFlow as described in the paper:

```text
CasFlow: Exploring Hierarchical Structures and Propagation Uncertainty for Cascade Prediction
Fan Zhou, Xovee Xu, Kunpeng Zhang, Siyuan Liu and Goce Trajcevski
Submitted for review
```

#### requirement
```
networkx==2.4
numpy==1.17.4
scikit-learn==0.21.3
scipy==1.3.3
tensorflow-gpu==2.0.0a0
```

#### parameters
```
learning rate :  0.0005
observation hour [7,19]
observation threshold :  1800
prediction time :  86400
cascade length [10,100]
```

#### quick run(default weibos)

```bash
cd models/CasFlow
python -u gen_cas.py 
python -u gen_emb.py 
python -u run.py 
```


### [CasCN](https://github.com/ChenNed/CasCN)

A TensorFlow implementation of Recurrent Cascades Convolution for the task of information cascades prediction, And the paper:

```text
X. Chen, F. Zhou, K. Zhang, G. Trajcevski, T. Zhong and F. Zhang, 
"Information Diffusion Prediction via Recurrent Cascades Convolution," 
2019 IEEE 35th International Conference on Data Engineering (ICDE),
 2019, pp. 770-781, doi: 10.1109/ICDE.2019.00074.
```

#### requirement

```
tensorflow-gpu==1.14.0
python==3.6
```
#### parameters

```
l2 0.001
learning rate :  0.0005
observation hour [7,19]
observation threshold :  1800
prediction time :  86400
cascade length [10,100]
```

#### quick run(default weibos)

```bash
cd models/CasCN
python -u gen_cas.py
python -u gen_graph_signal.py
python -u run.py
```


### [DeepCas](https://github.com/chengli-um/DeepCas)

A reference implementation of DeepCas as described in the paper:
```text
DeepCas: an End-to-end Predictor of Information Cascades
Cheng Li, Jiaqi Ma, Xiaoxiao Guo and Qiaozhu Mei
World wide web (WWW), 2017
```
The DeepCas algorithm learns the representation of cascade graphs in an end-to-end manner for cascade prediction.

#### requirement
```
tensorflow-gpu==1.14.0
gensim==4.0.1
networkx==2.5
```
#### parameters

```
dropout prob: 0.01
l2 1e-06
learning rate: 0.0005
emb_learning_rate: 0.005
observation hour [7,19]
observation threshold: 1800
prediction time: 86400
cascade length [10,100]
```
#### quick run(default weibos)

```bash
cd models/DeepCas
python -u gen_cas.py
python -u gen_walks.py
python -u gen_run.py
python -u run.py
```

### [TopoLSTM](http)

#### requirement
```
pytorch==^1.1.0
```
#### parameters

```
learning rate :  0.001
observation hour [7,19]
observation threshold :  1800
prediction time :  86400
cascade length [10,100]
```

#### quick run(default weibos)

```bash
cd models/TopoLSTM
python -u gen_cas.py
python -u gen_run.py
python -u run.py
```

## DataSet

Sina Weibo Dataset used in deephawkes paper,i.e., dataset_weibo.txt. It contains 119,313 messages in June 1, 2016. Each line contains the information of a certain message, the format of which is:

```text
<message_id>\tab<user_id>\tab<publish_time>\tab<retweet_number>\tab<retweets>
<message_id>:     the unique id of each message, ranging from 1 to 119,313.
<root_user_id>:   the unique id of root user. The user id ranges from 1 to 6,738,040.
<publish_time>:   the publish time of this message, recorded as unix timestamp.
<retweet_number>: the total number of retweets of this message within 24 hours.
<retweets>:       the retweets of this message, each retweet is split by " ". Within each retweet, it records 
the entile path for this retweet, the format of which is <user1>/<user2>/......<user n>:<retweet_time>.
```
