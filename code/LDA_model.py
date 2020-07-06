# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:02:57 2020

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
    
    