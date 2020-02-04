#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 13:53:59 2019

@author: xingwenpeng
"""


import ddr
import numpy as np
from gensim.models import KeyedVectors
import csv
import pandas as pd



#model_path='/Users/xingwenpeng/Desktop/csvdata/sgns.sogounews.bigram-char.bz2'
model_path='/Users/xingwenpeng/Desktop/csvdata/merge_sgns_bigram_char300.txt.bz2'

documents_path='/Users/xingwenpeng/Desktop/csvdata/docs.csv'

dictionary_directory='/Users/xingwenpeng/Desktop/nlp/expandedDic1700.csv'
#dictionary_directory3='/Users/xingwenpeng/Desktop/csvdata/words3.csv'
#dictionary_directory5='/Users/xingwenpeng/Desktop/csvdata/words.csv'
#dictionary_directory7='/Users/xingwenpeng/Desktop/csvdata/words7.csv'
#dictionary_directory9='/Users/xingwenpeng/Desktop/csvdata/words9.csv'

dictionary_vector_path='/Users/xingwenpeng/Desktop/csvdata/agg_dic_vectors.tsv'
#dictionary_vector_path3='/Users/xingwenpeng/Desktop/csvdata/agg_dic_vectors.tsv3'
#dictionary_vector_path5='/Users/xingwenpeng/Desktop/csvdata/agg_dic_vectors.tsv5'
#dictionary_vector_path7='/Users/xingwenpeng/Desktop/csvdata/agg_dic_vectors.tsv7'
#dictionary_vector_path9='/Users/xingwenpeng/Desktop/csvdata/agg_dic_vectors.tsv9'

document_vector_path='/Users/xingwenpeng/Desktop/csvdata/agg_doc_vecs.tsv'

document_loadings_out_path='/Users/xingwenpeng/Desktop/csvdata/document_dictionary_loadings.tsv'
#document_loadings_out_path3='/Users/xingwenpeng/Desktop/csvdata/document_dictionary_loadings3.tsv'
#document_loadings_out_path5='/Users/xingwenpeng/Desktop/csvdata/document_dictionary_loadings5.tsv'
#document_loadings_out_path7='/Users/xingwenpeng/Desktop/csvdata/document_dictionary_loadings7.tsv'
#document_loadings_out_path9='/Users/xingwenpeng/Desktop/csvdata/document_dictionary_loadings9.tsv'


#model = KeyedVectors.load_word2vec_format( model_path,encoding = "utf-8")
#model.save_word2vec_format("/Users/xingwenpeng/Desktop/csvdata/merge.model.bin", binary=True)

model, num_features, model_word_set=ddr.load_model("/Users/xingwenpeng/Desktop/csvdata/sougou.model.bin") 

#dic = open(dictionary_directory)

#dic_csv= csv.reader(dic)

#for i, rows in  enumerate(dic_csv):
#    model[rows]
#    print(model.most_similar(rows,topn=20))

dic_terms = ddr.terms_from_csv(input_path =dictionary_directory, delimiter ='\n'  )
#dic_terms3 = ddr.terms_from_csv(input_path =dictionary_directory3 ,delimiter ='\t' )
#dic_terms5 = ddr.terms_from_csv(input_path =dictionary_directory5 ,delimiter ='\t' )
#dic_terms7 = ddr.terms_from_csv(input_path =dictionary_directory7 ,delimiter ='\t' )
#dic_terms9 = ddr.terms_from_csv(input_path =dictionary_directory9 ,delimiter ='\t' )

#dic_terms_expand=np.loadtxt(dic_terms,skiprows=1)

agg_dic_vecs=ddr.dic_vecs(dic_terms=dic_terms, model=model, num_features=num_features, model_word_set=model_word_set)
#agg_dic_vecs3=ddr.dic_vecs(dic_terms=dic_terms3, model=model, num_features=num_features, model_word_set=model_word_set)
#agg_dic_vecs5=ddr.dic_vecs(dic_terms=dic_terms5, model=model, num_features=num_features, model_word_set=model_word_set)
#agg_dic_vecs7=ddr.dic_vecs(dic_terms=dic_terms7, model=model, num_features=num_features, model_word_set=model_word_set)
#agg_dic_vecs9=ddr.dic_vecs(dic_terms=dic_terms9, model=model, num_features=num_features, model_word_set=model_word_set)


ddr.write_dic_vecs(dic_vecs=agg_dic_vecs, output_path=dictionary_vector_path)
#ddr.write_dic_vecs(dic_vecs=agg_dic_vecs3, output_path=dictionary_vector_path3)
#ddr.write_dic_vecs(dic_vecs=agg_dic_vecs5, output_path=dictionary_vector_path5)
#ddr.write_dic_vecs(dic_vecs=agg_dic_vecs7, output_path=dictionary_vector_path7)
#ddr.write_dic_vecs(dic_vecs=agg_dic_vecs9, output_path=dictionary_vector_path9)

#ddr.doc_vecs_from_csv(input_path=dictionary_directory, output_path=dictionary_vector_path,model=model, num_features=num_features, model_word_set=model_word_set,text_col=0, delimiter = '\t',header=False)

ddr.doc_vecs_from_csv(input_path=documents_path, output_path=document_vector_path,model=model, num_features=num_features, model_word_set=model_word_set,text_col=0, delimiter = '\t',header=False)

ddr.get_loadings(agg_doc_vecs_path=document_vector_path,agg_dic_vecs_path=dictionary_vector_path,out_path=document_loadings_out_path,num_features=num_features)
#ddr.get_loadings(agg_doc_vecs_path=document_vector_path,agg_dic_vecs_path=dictionary_vector_path3,out_path=document_loadings_out_path3,num_features=num_features)
#ddr.get_loadings(agg_doc_vecs_path=document_vector_path,agg_dic_vecs_path=dictionary_vector_path5,out_path=document_loadings_out_path5,num_features=num_features)
#ddr.get_loadings(agg_doc_vecs_path=document_vector_path,agg_dic_vecs_path=dictionary_vector_path7,out_path=document_loadings_out_path7,num_features=num_features)
#ddr.get_loadings(agg_doc_vecs_path=document_vector_path,agg_dic_vecs_path=dictionary_vector_path9,out_path=document_loadings_out_path9,num_features=num_features)


'''
def cosVector(x,y):
    if(len(x)!=len(y)):
        print('error input,x and y is not in the same space')
        return;
    result1=0.0;
    result2=0.0;
    result3=0.0;
    for i in range(len(x)):
        result1+=x[i]*y[i]   #sum(X*Y)
        result2+=x[i]**2     #sum(X*X)
        result3+=y[i]**2    #sum(Y*Y)
        
    a=result1/((result2*result3)**0.5)
    print("result is "+str(a))
    return a

def outPut(dictionary_vector_path,document_vector_path,excel_out_put_path,i,n):    
  f1 = open(dictionary_vector_path)
  case_train=np.loadtxt(f1,skiprows=1)
  f1.close()
  matrix1=np.array(case_train)

  f1 = open(document_vector_path)
  data= np.loadtxt(document_vector_path,skiprows=1)
  f1.close()
  matrix2=np.array(data)
  matrix2 = np.delete(matrix2, 0, axis=1)
  similarity=[]
  for i in range(matrix2.shape[0]): #获得文章的个数
     consin=cosVector(matrix1,matrix2[i]) #计算字典与各个文章的相似度
     similarity.append(consin)  
  # list转dataframe
  df = pd.DataFrame(similarity, columns=['similarity'+i])   
    # 保存到本地excel
  df.to_excel(excel_out_put_path, index=False,startcol=n)
  
  
  
  
  
excel_out_put_path="/Users/xingwenpeng/Desktop/csvdata/similarity"
excel_out_put_path3="/Users/xingwenpeng/Desktop/csvdata/similarity"
excel_out_put_path5="/Users/xingwenpeng/Desktop/csvdata/similarity"
excel_out_put_path7="/Users/xingwenpeng/Desktop/csvdata/similarity"
excel_out_put_path9="/Users/xingwenpeng/Desktop/csvdata/similarity"
outPut(dictionary_vector_path,document_vector_path,excel_out_put_path,1,0)
outPut(dictionary_vector_path3,document_vector_path,excel_out_put_path3,3,1)
outPut(dictionary_vector_path5,document_vector_path,excel_out_put_path5,5,2)
outPut(dictionary_vector_path7,document_vector_path,excel_out_put_path7,7,3)
outPut(dictionary_vector_path9,document_vector_path,excel_out_put_path9,9,4)
'''