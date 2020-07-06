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
import numpy as np
import nltk
from PIL import Image                     
from wordcloud import WordCloud, ImageColorGenerator,STOPWORDS
import bcolz
import pickle

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

def paint_wordcloud(text, pic_name, outfile):
    image = Image.open(r'cloud.jpg')
    graph = np.array(image)
    wc = WordCloud(font_path=r"C:\Windows\Fonts\times.ttf",
                   background_color='white',
                   width=400,
                   height=300,
                   stopwords = stops,
                   max_font_size=200,
                   max_words=100)#,min_font_size=10)#,mode='RGBA',colormap='pink')
    wc.generate(text)
    #image_color = ImageColorGenerator(graph)
    #wc.recolor(color_func = image_color)
    wc.to_file(outfile)
    plt.figure(pic_name) 
    plt.imshow(wc)     
    plt.axis("off")     
    plt.show()
    
if __name__=='__main__':
    
    context_dic = {}#{doi:TI AB,DT,SC,AF}
    refer_dic = {} #{doi:doi}
    time_dic = {} #{doi:PY}
    doi_index = [] #doi
    
    print('\n' + '数据预处理' + '\n')
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
            
    #文章类别分析  
    print('\n' + '文章类别分析' + '\n')
    DT = {}
    for key in context_dic.keys():
        if context_dic[key][2] in DT.keys():
            DT[context_dic[key][2]] = DT[context_dic[key][2]] + 1
        else:
            DT[context_dic[key][2]] = 1
            
    DT_ls = sorted(DT.items(), key=lambda x: x[1], reverse=True)
    
    name_list = []  
    num_list = []  
    for ele in DT_ls:
        name_list.append(ele[0])
        num_list.append(ele[1])
        
    name_list = name_list[:3] + ['others']
    num_list = num_list[:3] + [sum(num_list[3:])]
    
    plt.pie(num_list,labels=name_list,autopct='%1.2f%%',pctdistance = 0.6)
    plt.title('Category percentage')
    plt.savefig('Category_percentage.jpg')
    plt.show()
    
    
    # 学科类别分析
    print('\n' + '学科类别分析' + '\n')
    SC = {}
    for key in context_dic.keys():
        if context_dic[key][3] in SC.keys():
            SC[context_dic[key][3]] = SC[context_dic[key][3]] + 1
        else:
            SC[context_dic[key][3]] = 1
    SC_ls = sorted(SC.items(), key=lambda x: x[1], reverse=True)
    
    name = []  
    num = []  
    for ele in SC_ls:
        name.append(ele[0])
        num.append(ele[1])
        
    name_list = name[:5]
    num_list = num[:5]
    
    plt.barh(name_list, num_list,color="blue")
    plt.xlabel('frequency')
    plt.ylabel('subject category')
    plt.title('TOP 5 subject category')
    plt.savefig('subject_category.jpg')
    plt.show()
    
    # 每篇文章引用数据库文章数目
    print('\n' + '每篇文章引用数据库文章数目' + '\n')
    CR = {}
    for key in refer_dic.keys():
        if len(refer_dic[key]) in CR.keys():
            CR[len(refer_dic[key])] = CR[len(refer_dic[key])] + 1
        else:
            CR[len(refer_dic[key])] = 1
                                    
    CR_ls = sorted(CR.items(), key=lambda x: x[0], reverse=False)
    
    x_list = []  
    y_list = []  
    for ele in CR_ls:
        x_list.append(ele[0])
        y_list.append(ele[1])
        
    plt.bar(x_list, y_list, color="blue")
    plt.xlabel('Reference number')
    plt.ylabel('frequency')
    plt.title('The frequency of Reference number')
    plt.savefig('Reference_number.jpg')
    plt.show()
    
    # 使用词云分析每五年研究热点
    print('\n' + '使用词云分析每五年研究热点' + '\n')
    stops = set(list(STOPWORDS) + ['change','changes','climate',
                               'model','global','response',
                               'increase','effect','impact',
                               'based','using','many','data',
                               'result','show','effects','impacts','may'])
    ls1 = [] #2000-2005
    ls2 = [] #2006-2010
    ls3 = [] #2011-2015
    ls4 = [] #2016-2020
    
    t1 = ['2000', '2001', '2002', '2003', '2004', '2005']
    t2 = ['2006', '2007', '2008', '2009', '2010']
    t3 = ['2011', '2012', '2013', '2014', '2015']
    t4 = ['2016', '2017', '2018', '2019', '2020']
    for key in time_dic.keys():
        year = time_dic[key]
        if year in t1:
            ls1.append(context_dic[key][0] + context_dic[key][1])
        elif year in t2:
            ls2.append(context_dic[key][0] + context_dic[key][1])
        elif year in t3:
            ls3.append(context_dic[key][0] + context_dic[key][1])
        elif year in t4:
            ls4.append(context_dic[key][0] + context_dic[key][1])
            
    paint_wordcloud(' '.join(ls1), '2000-2005 hot topic', 't1.jpg')
    
    print('\n' + '准备glove词向量' + '\n')
    words = ['<unk>']
    idx = 1
    word2idx = {}
    vectors = bcolz.carray(np.zeros(1), rootdir=f'6B.50.dat', mode='w')
    word2idx['<unk>'] = 0
    with open(f'glove.6B.50d.txt', 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    
    vectors = bcolz.carray(vectors[1:].reshape((400000, 50)), rootdir=f'6B.50.dat', mode='w')
    vectors.flush()
    pickle.dump(words, open(f'6B.50_words.pkl', 'wb'))
    pickle.dump(word2idx, open(f'6B.50_idx.pkl', 'wb'))
    