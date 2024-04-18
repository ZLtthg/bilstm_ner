import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
    
#extract train, dev, and test data
train_data_path = 'train.conll'
dev_data_path = 'dev.conll'
test_data_path = 'test.conll'
train_data = pd.read_csv(train_data_path,sep='\t',header=None,skiprows=[0],names=['word','pos','chunk','tag'],quoting=3)
dev_data = pd.read_csv(dev_data_path,sep='\t',header=None,skiprows=[0],names=['word','pos','chunk','tag'],quoting=3)
test_data = pd.read_csv(test_data_path,sep='\t',header=None,skiprows=[0],names=['word','pos','chunk','tag'],quoting=3)
train_data_array = train_data.to_numpy()
dev_data_array = train_data.to_numpy()
test_data_array = train_data.to_numpy()


#extract embedding and from word_to_index glove.6B.50d.txt
def embedd_pro():
    embeddings = []
    word_2_index = {}
    with open('glove.6B.50d.txt','rt',errors='ignore') as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        if(i_word in word_2_index):
            pass
        else:
            word_2_index[i_word] = len(word_2_index)
            embeddings.append(i_embeddings)
    embeddings = np.array(embeddings)  
    word_2_index['<UNK>'] = len(word_2_index)
    word_2_index['<PAD>'] = len(word_2_index)
    embeddings = np.vstack((embeddings,np.zeros((1,embeddings.shape[1])),np.mean(embeddings,axis=0,keepdims=True) ))
    return embeddings,word_2_index

#get the embedding and word_to_index 
embeddings,word_2_index = embedd_pro()

#return word_lists and tag_lists and tag_2_index(when processing training data).For Example
#word_lists:[['EU','rejects','German','call','to'],['The','European','Commission','said']]
#tag_lists:[['B-ORG','O','B-MISC','O'],['O','B-ORG','I-ORG','O']]
#tag_2_index:{'O':0,'B-PER':1,'I-PER':2,'B-ORG':3,'I-ORG':4,'B-MISC':5,'I-MISC':6,'B-LOC':7,'I-LOC':8,'<PAD>':9,'<UNK>':10}
def build_corpus(data_array,make_vocab=True):
    word_lists = []
    tag_lists = []
    word_list = []
    tag_list = []
    for i in range(len(train_data_array)):
        if(str(train_data_array[i][3])!='nan'):
            word_list.append(str(train_data_array[i][0]))
            tag_list.append(train_data_array[i][3])
        else:
            word_lists.append(word_list)
            tag_lists.append(tag_list)
            word_list = []
            tag_list = []

    tag_2_index = {
      'O':0,
      'B-PER':1,
      'I-PER':2,
      'B-ORG':3,
      'I-ORG':4,
      'B-MISC':5,
      'I-MISC':6,
      'B-LOC':7,
      'I-LOC':8,
      '<PAD>':9,
      '<UNK>':10
    }
    if make_vocab:
        return word_lists,tag_lists,tag_2_index
    else:
        return word_lists,tag_lists

#Create a dataset
class MyDataset(Dataset):
    def __init__(self,datas,tags,word_2_index,tag_2_index):
        self.datas = datas
        self.tags = tags
        self.word_2_index = word_2_index
        self.tag_2_index = tag_2_index

    def __getitem__(self,index):
        data = self.datas[index]
        tag = self.tags[index]
        data_index = [self.word_2_index.get(i.lower(),self.word_2_index["<UNK>"]) for i in data]
        tag_index = [self.tag_2_index[i] for i in tag]
        return data_index,tag_index


    def __len__(self):
        assert len(self.datas) == len(self.tags)
        return len(self.datas)

    def batch_data_pro(self,batch_datas):
        global device
        data , tag = [],[]
        da_len = []
        for da,ta in batch_datas:
            data.append(da)
            tag.append(ta)
            da_len.append(len(da))
        max_len = max(da_len)

        data = [i + [self.word_2_index["<PAD>"]] * (max_len - len(i))     for i in data]
        tag = [i + [self.tag_2_index["<PAD>"]] * (max_len - len(i))     for i in tag]

        data = torch.tensor(data,dtype=torch.long,device = device)
        tag = torch.tensor(tag,dtype=torch.long,device = device)
        return data , tag, da_len

#Create a ModelClass
class MyModel(nn.Module):
    def __init__(self,embedding_num,hidden_num,corpus_num,class_num,pad_index):
        super().__init__()
        self.embedding_num = embedding_num
        self.hidden_num = hidden_num
        self.corpus_num = corpus_num

        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(embeddings).float())
        self.lstm = nn.LSTM(embedding_num,hidden_num,batch_first=True,bidirectional=True)
        self.classifier = nn.Linear(hidden_num*2,class_num)
        self.cross_loss = nn.CrossEntropyLoss(ignore_index=pad_index)

    def forward(self,data_index,data_len , tag_index=None):

        em = self.embedding(data_index)
        pack = nn.utils.rnn.pack_padded_sequence(em,data_len,batch_first=True)
        output,_ = self.lstm(pack)
        output,lens = nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
        pre = self.classifier(output)

        self.pre = torch.argmax(pre, dim=-1).reshape(-1)

        if tag_index is not None:
            loss = self.cross_loss(pre.reshape(-1,pre.shape[-1]),tag_index.reshape(-1))
            return loss


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"


    train_word_lists,train_tag_lists,tag_2_index = build_corpus(train_data_array,make_vocab=True)
    word_2_index = word_2_index

    dev_word_lists, dev_tag_lists = build_corpus(dev_data_array,make_vocab=False)
    test_word_lists,test_tag_lists = build_corpus(test_data_array,make_vocab=False)

    #parameters
    corpus_num = len(word_2_index)
    class_num = len(tag_2_index)
    train_batch_size = 1
    dev_batch_size = 1
    epoch = 20
    lr = 0.001
    embedding_num = 50
    hidden_num    = 100


    train_dataset = MyDataset(train_word_lists,train_tag_lists,word_2_index, tag_2_index)
    train_dataloader = DataLoader(train_dataset,batch_size=train_batch_size,shuffle=False,collate_fn=train_dataset.batch_data_pro)

    dev_dataset = MyDataset(dev_word_lists, dev_tag_lists, word_2_index, tag_2_index)
    dev_dataloader = DataLoader(dev_dataset, batch_size=dev_batch_size, shuffle=False,collate_fn=dev_dataset.batch_data_pro)

    test_dataset = MyDataset(test_word_lists, test_tag_lists, word_2_index, tag_2_index)
    test_dataloader = DataLoader(test_dataset, batch_size=dev_batch_size, shuffle=False,collate_fn=dev_dataset.batch_data_pro)


    model = MyModel(embedding_num,hidden_num,corpus_num,class_num,word_2_index["<PAD>"])
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(),lr = lr)

    f1_scores_dev_data = []
    for e in range(epoch):
        #train model
        model.train()
        for data , tag, da_len in train_dataloader:
            loss = model.forward(data,da_len,tag)
            loss.backward()
            opt.step()
            opt.zero_grad()
        #evaluate model
        model.eval() 
        evl_all_tag = []
        evl_all_pre = []
        for dev_data , dev_tag, dev_da_len in dev_dataloader:
            test_loss = model.forward(dev_data,dev_da_len,dev_tag)
            evl_all_tag.extend(dev_tag.reshape(-1).cpu().numpy().tolist())
            evl_all_pre.extend(model.pre.cpu().numpy().tolist())

        score = f1_score(evl_all_pre,evl_all_tag,average='macro')
        f1_scores_dev_data.append(score)
        print('epoch',e,',  Macor-Average on devdata: ',score)

    
    

    #test model
    model.eval()
    test_all_tag = []
    test_all_pre = []
    for test_data,test_tag,test_da_len in test_dataloader:
        test_loss = model.forward(test_data,test_da_len,test_tag)
        test_all_tag.extend(test_tag.reshape(-1).cpu().numpy().tolist())
        test_all_pre.extend(model.pre.cpu().numpy().tolist())
       
    score = f1_score(test_all_pre,test_all_tag,average='macro')
    print('Macor-Average on testdata: ',score)  





    plt.plot(list(range(epoch)),f1_scores_dev_data)
    plt.xlabel('epoch')
    plt.ylabel('F1-Macro-Scores')
    plt.title('NER-Result')
    plt.show()