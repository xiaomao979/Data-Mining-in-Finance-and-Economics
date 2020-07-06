# -*- coding: utf-8 -*-
"""
Created on Sun May 31 21:23:55 2020

@author: yuansiyu
"""

import re
import copy
import random
import matplotlib.pyplot as plt
import pylab as pl
from nltk.corpus import stopwords
from gensim import corpora,models
import numpy as np
import numpy as np
import bcolz
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def Construction_context(address, context_dic, time_dic, doi_index):
    f = open(address, encoding='UTF-8')
    line = f.readline()
    dic = {}
    while line:
        
        if line[:2] == 'DI':
            doi = line[2:]
            doi = doi.strip()
            dic['doi'] = doi

            
        if line[:2] == 'TI':
            title = line[2:]
            title = title.strip()
            dic['TI'] = title
                        
        if line[:2] == 'AB':
            ab = line[2:]
            ab = ab.strip()
            dic['AB'] = ab
            
        if line[:2] == 'DT':
            cate = line[2:]
            cate = cate.strip()
            dic['DT'] = cate

        if line[:2] == 'SC':
            sc = line[2:]
            sc = sc.strip()
            dic['SC'] = sc
            
        if line[:2] == 'AF':
            author = line[2:]
            author = author.strip()
            dic['AF'] = author
            
        if line[:2] == 'PY':
            date = line[2:]
            date = date.strip()
            dic['PY'] = date
        
        line = f.readline()
        
    context_dic[dic['doi']] = [dic['TI'],dic['AB'],dic['DT'],dic['SC'],dic['AF']]
    doi_index.append(dic['doi'])
    if 'PY' not in dic.keys():
        time_dic[dic['doi']] = address[5:9]
    else:    
        time_dic[dic['doi']] = dic['PY']
    
    f.close()
    
def Construction_reference(address, refer_dic):
    f = open(address, encoding='UTF-8')
    text = f.read()
    text_ls = text.split('\n')
    for i in range(len(text_ls)):
        if text_ls[i][:2] == 'CR':
            begin_index = i
            break

    for i in range(begin_index+2, len(text_ls)):
        if text_ls[i][0] != ' ':
            end_index = i-1
            break

    r = []
    for i in range(begin_index, end_index+1):
        try:
            index = re.search('DOI', text_ls[i]).span()
            refer_doi = text_ls[i][index[1]:]
            refer_doi = refer_doi.strip()
            r.append(refer_doi)
        except:
            continue
    
    for i in range(len(text_ls)):
        if text_ls[i][:2] == 'DI':
            doi = text_ls[i][2:]
            doi = doi.strip()

    refer_dic[doi] = r
    f.close()

def clean(doc):
    filtered_words = [word for word in doc if word not in stop]
    return filtered_words

def check(x,y):
    for ele in x:
        if ele not in y:
            return 0
    
    for ele in y:
        if ele not in x:
            return 0
    
    return 1

class ReferenceDataset(Dataset):
    def __init__(self, docs, word2idx, max_text, max_DT, max_SC):
        super(ReferenceDataset,self).__init__()
        
        self.docs = docs
        self.word2idx = word2idx
    
    def __len__(self):#这个数据集一共有多少个item
        return len(self.docs)
    
    def change(self, doc):
        text = (doc[0] + doc[1]).lower().split()
        text_encode = [self.word2idx.get(word, self.word2idx['<unk>']) for word in text]
        delta = max_text - len(text_encode)+1
        text_encode = text_encode + [0 for i in range(delta)]
        text_encode = np.array(text_encode) 
        text_encode = torch.from_numpy(text_encode)
        
        
        doc_DT = doc[2].lower().split()
        doc_DT_encode = [self.word2idx.get(word, self.word2idx['<unk>']) for word in doc_DT]
        delta = max_DT - len(doc_DT_encode)+1
        doc_DT_encode = doc_DT_encode + [0 for i in range(delta)]
        doc_DT_encode = np.array(doc_DT_encode) 
        doc_DT_encode = torch.from_numpy(doc_DT_encode)
        
        doc_SC = doc[3].lower().split()
        doc_SC_encode = [self.word2idx.get(word, self.word2idx['<unk>']) for word in doc_SC]
        delta = max_SC - len(doc_SC_encode)+1
        doc_SC_encode = doc_SC_encode + [0 for i in range(delta)]
        doc_SC_encode = np.array(doc_SC_encode) 
        doc_SC_encode = torch.from_numpy(doc_SC_encode)
        
        doc_TO = doc[5]
        doc_TO_encode = [self.word2idx.get(word.lower(), self.word2idx['<unk>']) for word in doc_TO]
        doc_TO_encode = np.array(doc_TO_encode)
        doc_TO_encode = torch.from_numpy(doc_TO_encode)
        
        return text_encode, doc_DT_encode, doc_SC_encode, doc_TO_encode
    def __getitem__(self, idx):#给定一个index返回一个item

        sample = self.docs[idx]
        doc = sample[0]
        cr = sample[1]
        label = torch.Tensor([sample[2]])

        doc_text, doc_DT, doc_SC, doc_TO = self.change(doc)
        cr_text, cr_DT, cr_SC, cr_TO = self.change(cr)

        return doc_text, doc_DT, doc_SC, doc_TO, cr_text, cr_DT, cr_SC, cr_TO, label

def binary_accuracy(preds,y):
    rounded_preds = torch.round(torch.sigmoid(preds))#0,1prob
    correct = (rounded_preds == y).float()
    acc = correct.sum()/len(correct)
    return acc

def train(model, dataloader, optimizer, loss_fn):
    epoch_loss, epoch_acc = 0.,0.
    model.train()
    total_len = 0.
    for i, (doc_text, doc_DT, doc_SC, doc_TO, cr_text, cr_DT, cr_SC, cr_TO, label) in enumerate(dataloader):
        doc_text, doc_DT, doc_SC, doc_TO = doc_text.long().to(device), doc_DT.long().to(device), doc_SC.long().to(device), doc_TO.long().to(device)
        cr_text, cr_DT, cr_SC, cr_TO = cr_text.long().to(device), cr_DT.long().to(device), cr_SC.long().to(device), cr_TO.long().to(device)
        label = label.to(device)
        preds = model(doc_text, doc_DT, doc_SC, doc_TO, cr_text, cr_DT, cr_SC, cr_TO).squeeze()
        label = label.squeeze()
        
        #print(preds)
        #print(label)
        
        loss = loss_fn(preds, label)
        acc = binary_accuracy(preds, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * len(label)
        epoch_acc += acc.item() * len(label)
        total_len += len(label)
        
    return epoch_loss / total_len, epoch_acc / total_len

def evaluate(model, dataloader, loss_fn):
    epoch_loss, epoch_acc = 0., 0.
    model.eval()
    total_len = 0.

    for i, (doc_text, doc_DT, doc_SC, doc_TO, cr_text, cr_DT, cr_SC, cr_TO, label) in enumerate(dataloader):
        doc_text, doc_DT, doc_SC, doc_TO = doc_text.long().to(device), doc_DT.long().to(device), doc_SC.long().to(device), doc_TO.long().to(device)
        cr_text, cr_DT, cr_SC, cr_TO = cr_text.long().to(device), cr_DT.long().to(device), cr_SC.long().to(device), cr_TO.long().to(device)
        label = label.to(device)
        preds = model(doc_text, doc_DT, doc_SC, doc_TO, cr_text, cr_DT, cr_SC, cr_TO).squeeze()
        label = label.squeeze()
       
        loss = loss_fn(preds, label)
        acc = binary_accuracy(preds, label)
        
        epoch_loss += loss.item() * len(label)
        epoch_acc += acc.item() * len(label)
        total_len += len(label)
        
    model.train()
    return epoch_loss / total_len, epoch_acc / total_len

class MyNet(nn.Module):
    def __init__(self, batch_size, weights_matrix, max_text, max_DT, max_SC, feature_size, non_trainable = True):
        super(MyNet, self).__init__()
        num_embeddings, embedding_dim = weights_matrix.shape
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        self.embed.weight.data.copy_(torch.from_numpy(weights_matrix))
        if non_trainable:
            self.embed.weight.requires_grad = False
        #self.embedding, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        #self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.conv1 = nn.Conv1d(in_channels = embedding_dim, out_channels = feature_size, kernel_size = 5)
        self.conv2 = nn.Conv1d(in_channels = embedding_dim, out_channels = feature_size, kernel_size = 4)
        self.conv3 = nn.Conv1d(in_channels = embedding_dim, out_channels = feature_size, kernel_size = 3)
        
        self.conv4 = nn.Conv1d(in_channels = embedding_dim, out_channels = feature_size, kernel_size = 1)        
        self.conv5 = nn.Conv1d(in_channels = embedding_dim, out_channels = feature_size, kernel_size = 1)        
        self.conv6 = nn.Conv1d(in_channels = embedding_dim, out_channels = feature_size, kernel_size = 1)
        
        self.pool1 = nn.MaxPool1d(max_text+1 - 5 + 1)
        self.pool2 = nn.MaxPool1d(max_text+1 - 4 + 1)
        self.pool3 = nn.MaxPool1d(max_text+1 - 3 + 1)
        
        self.pool4 = nn.MaxPool1d(max_DT+1 - 1 + 1)
        self.pool5 = nn.MaxPool1d(max_SC+1 - 1 + 1)
        self.pool6 = nn.MaxPool1d(5 - 1 + 1)
        
        self.dropout = nn.Dropout(p=0.2)
        
        self.simi_weight = nn.Parameter(torch.zeros(batch_size, feature_size * 3, feature_size * 3))
        
        self.fc1 = nn.Linear(6*feature_size + 1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, d_text, doc_DT, doc_SC, doc_TO, c_text, cr_DT, cr_SC, cr_TO):
        doc_text = self.embed(d_text).permute(0, 2, 1) #torch.Size([32, 50, 575])
        cr_text = self.embed(c_text).permute(0, 2, 1)
        #doc_DT_embed = self.embed(doc_DT).permute(0, 2, 1)
        #doc_SC_embed = self.embed(doc_SC).permute(0, 2, 1)
        #doc_TO_embed = self.embed(doc_TO).permute(0, 2, 1)
        #cr_DT_embed = self.embed(cr_DT).permute(0, 2, 1)
        #cr_SC_embed = self.embed(cr_SC).permute(0, 2, 1)
        #cr_TO_embed = self.embed(cr_TO).permute(0, 2, 1)

        d1 = self.pool1(F.relu(self.conv1(doc_text))) #1 torch.Size([32, 100, 571]) -> torch.Size([32, 100, 1])
        d2 = self.pool2(F.relu(self.conv2(doc_text))) #1
        d3 = self.pool3(F.relu(self.conv3(doc_text))) #1
        
        #d_DT = self.pool4(F.relu(self.conv4(doc_DT_embed))) #4 torch.Size([32, 100, 1])
        #d_SC = self.pool5(F.relu(self.conv5(doc_SC_embed)))#13 torch.Size([32, 100, 1])
        #d_TO = self.pool6(F.relu(self.conv6(doc_TO_embed))) #5 torch.Size([32, 100, 1])
        

        
        c1 = self.pool1(F.relu(self.conv1(cr_text)))
        c2 = self.pool2(F.relu(self.conv2(cr_text)))
        c3 = self.pool3(F.relu(self.conv3(cr_text)))
        
        
        #c_DT = self.pool4(F.relu(self.conv4(cr_DT_embed))) #4
        #c_SC = self.pool5(F.relu(self.conv5(cr_SC_embed))) #13
        #c_TO = self.pool6(F.relu(self.conv6(cr_TO_embed))) #5

        
        #DT = torch.mul(d_DT, c_DT) #4
        #SC = torch.mul(d_SC, c_SC) #13
        #TO = torch.mul(d_TO, c_TO) #5
    
        
        Q = self.dropout(torch.cat((d1, d2, d3), 1).squeeze(2))# torch.Size([32, 300])
        A = self.dropout(torch.cat((c1, c2, c3), 1).squeeze(2))# torch.Size([32, 300])
        
        

        transform_Q = torch.cat((d1, d2, d3), 1).permute(0, 2, 1) # torch.Size([32, 1, 300])

        temp = torch.bmm(transform_Q, self.simi_weight) # torch.Size([32, 1, 300])
        transform_C = torch.cat((c1, c2, c3), 1)
        sim = torch.bmm(temp, transform_C).squeeze(2) # torch.Size([32, 1, 1]) -> torch.Size([32, 1])
        #info = torch.cat((d1, d2, d3, c1, c2, c3, DT, SC, TO), 1) # torch.Size([32, 900, 1])
        #info = torch.cat((d1, d2, d3, c1, c2, c3), 1) # torch.Size([32, 900, 1])
        info = torch.cat((Q, sim, A), 1) # torch.Size([32, 900])
        info1 = self.fc1(info)

        info2 = self.fc2(info1)

        info3 = self.fc3(info2)

        return info3

if __name__=='__main__':
    
    context_dic = {}#{doi:TI AB,DT,SC,AF}
    refer_dic = {} #{doi:doi}
    time_dic = {} #{doi:PY}
    doi_index = [] #doi
    
    count = [107,117,132,152,164,191,212,271,325,381,455,521,556,630,680,784,876,945,1041,1093,345]
    for i in range(21):
        if i < 10:
            index = '0' + str(i)
        else:
            index = str(i)
        for j in range(count[i]):
            address = "amcl/"+ "20" + index + "/" + "cl_" + "20" + index + "_" + str(j+1) + ".txt"
            try:
                Construction_context(address, context_dic, time_dic, doi_index)
            except:
                continue
            
    count = [107,117,132,152,164,191,212,271,325,381,455,521,556,630,680,784,876,945,1041,1093,345]
    for i in range(21):
        if i < 10:
            index = '0' + str(i)
        else:
            index = str(i)
        for j in range(count[i]):
            address = "amcl/"+ "20" + index + "/" + "cl_" + "20" + index + "_" + str(j+1) + ".txt"
            try:
                Construction_reference(address, refer_dic)
            except:
                continue
            
            
    co_refer = refer_dic.copy()
    for key in co_refer.keys():
        ls = []
        for ele in refer_dic[key]:
            if ele in context_dic.keys():
                ls.append(ele)
        if len(ls) == 0:
            refer_dic.pop(key)
        else:
            refer_dic[key] = ls
    
    co_context = context_dic.copy()
    for key in co_context.keys():
        if key not in refer_dic.keys():
            context_dic.pop(key)
    
    co_refer = refer_dic.copy()
    for key in co_refer.keys():
        if key not in context_dic.keys():
            refer_dic.pop(key)
            
    co_time = time_dic.copy()
    for key in co_time.keys():
        if key not in context_dic.keys():
            time_dic.pop(key)
    
    co_doi = []
    for i in range(len(doi_index)):
        if doi_index[i] in context_dic.keys():
            co_doi.append(doi_index[i])
            
    doc_complete = []
    for key in context_dic.keys():
        TI = context_dic[key][0]
        AB = context_dic[key][1]
        doc = TI + '. ' + AB
        doc_complete.append(doc)
        
    stop = stopwords.words('english')
    doc_clean = [clean(doc.split()) for doc in doc_complete]
    dictionary = corpora.Dictionary(doc_clean)
    corpus = [dictionary.doc2bow(doc) for doc in doc_clean]
    corpus_tfidf = models.TfidfModel(corpus)[corpus]
    num_topic = 10
    
    lda = models.LdaModel(corpus_tfidf, num_topics = num_topic,
                      id2word = dictionary, alpha = 'auto',
                      eta = 'auto', minimum_probability = 0.001)
    num_show_term = 5
    topic_dic = {}
    for topic_id in range(num_topic):
        print('\n'+ 'topic ' + str(topic_id) + '\n')
        term_distribute_all = lda.get_topic_terms(topicid = topic_id)
        term_distribute = term_distribute_all[:num_show_term]
        term_distribute = np.array(term_distribute)
        term_id = term_distribute[:,0].astype(np.int)
    
        ls = []
        for t in term_id:
            ls.append(dictionary.id2token[t])
            print(dictionary.id2token[t])
        topic_dic[topic_id] = ls
    
    num_show_topic = 1
    topics = lda.get_document_topics(corpus_tfidf)
    topic_index = []
    for i in range(len(doc_complete)):
        topic = np.array(topics[i])
        topic_distribute = np.array(topic[:,1])
        topic_idx = topic_distribute.argsort()[:-num_show_topic-1:-1]
        topic_index.append(topic_idx)
        
    doc_topic = [doc_t for doc_t in lda[corpus_tfidf]]
    i = 0
    x = context_dic.copy()
    for key in context_dic.keys():
        context_dic[key].append(topic_dic[topic_index[i][0]])
        i = i + 1
    
    vectors = bcolz.open(f'6B.50.dat')[:]
    words = pickle.load(open(f'6B.50_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'6B.50_idx.pkl', 'rb'))
        
    glove = {}
    for w in words[1:]:
        try:
            glove[w] = vectors[word2idx[w]-1]
        except:
            continue
    
    matrix_len = len(word2idx)
    weights_matrix = np.zeros((matrix_len, 50))
    words_found = 0
    
    for word, i in word2idx.items():
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(50,))
    
    batch_size = 32
    
    co_refer = refer_dic.copy()
    for key in co_refer.keys():
        ls = []
        for ele in refer_dic[key]:
            if ele in context_dic.keys():
                ls.append(ele)
        if len(ls) == 0:
            refer_dic.pop(key)
        else:
            refer_dic[key] = ls
            
    do_copy = []
    
    for ele in doi_index:
        if ele in context_dic.keys():
            do_copy.append(ele)
            
    doi_index = do_copy
    length = len(doi_index)
    dat = []
    for doc in refer_dic:
        k = len(refer_dic[doc])
        for cr in refer_dic[doc]:
            doc_sample = []
            doc_sample.append(context_dic[doc])
            doc_sample.append(context_dic[cr])
            doc_sample.append(1)
            dat.append(doc_sample)
        for i in range(k):
            doc_sample = []
            r =  random.randint(0, length-1)
            while doi_index[r] in refer_dic[doc]:
                r =  random.randint(0, length-1)
            neg = doi_index[r]
            doc_sample.append(context_dic[doc])
            doc_sample.append(context_dic[neg])
            doc_sample.append(0)
            dat.append(doc_sample)
    
    max_text = 0 
    max_DT = 0
    max_SC = 0
    max_TO = 0
    for key in context_dic.keys():
        ls = context_dic[key]
        text = ls[0]+ls[1]
        len_text = len(text.split())
        if len_text > max_text:
            max_text = len_text
        DT = ls[2]
        len_DT = len(DT.split())
        if len_DT > max_DT:
            max_DT = len_DT
        SC = ls[3]
        len_SC = len(SC.split())
        if len_SC > max_SC:
            max_SC = len_SC
        if len(ls[5]) > max_TO:
            max_TO = len(ls[5])
    
    random.shuffle(dat)
    train_data = dat[:96000]
    test_data = dat[96000:128352]
    
    corr = 0
    for ele in test_data:
        flag = check(ele[0][5],ele[1][5])
        if flag == ele[2]:
            corr = corr + 1
    
    print('accuracy of baseline is {}'.format(corr/len(test_data)))
    
    train_dataset = ReferenceDataset(train_data, word2idx, max_text, max_DT, max_SC)
    test_dataset = ReferenceDataset(test_data, word2idx, max_text, max_DT, max_SC)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 0)
    
    feature_size = 100
    model = MyNet(batch_size, weights_matrix, max_text, max_DT, max_SC, feature_size, non_trainable = False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.BCEWithLogitsLoss()
    
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    num_epochs = 30
    best_test_acc = 0.
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader,optimizer, loss_fn)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'economy_model.pth')
        
        print("epoch",epoch,"train:",train_loss, train_acc)
        print("epoch",epoch,"test:",test_loss, test_acc)
        
            