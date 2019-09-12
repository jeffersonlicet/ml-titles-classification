import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import WordPunctTokenizer
import pickle
from tqdm import tqdm
from utils_hash import hash_tokens
from multiprocessing import Pool

spanishSnow = SnowballStemmer('spanish')
portugueseSnow = SnowballStemmer('portuguese')

dictionary = np.load('./dic.npy', allow_pickle=True)

f = open('./embeddings/es-fast-300.txt')
es = f.readlines()
f.close()
es = np.array(es)
es = es[1:int(es[0].split()[0])]

#es = es[10:30]

f = open('./embeddings/pt-fast-300.txt')
pt = f.readlines()
f.close()
pt = np.array(pt)
pt = pt[1:int(pt[0].split()[0])]
#pt = pt[10:30]

chunks_pt = np.array_split(pt, 8)
chunks_es = np.array_split(es, 8)

vocabulary_size = dictionary.shape[0] + 1

def _hash(lines):
  _dic = dict()
  for line in tqdm(lines):
    values = line.split()
    try:
      word = spanishSnow.stem(values[0])
      if word in dictionary:
        coefs = np.asarray(values[1:], dtype='float32')
        _dic[word] = coefs
        #print("Spanish: " + word)
    except Exception as e:
      print(e)
      print("Error on line: ")
  return _dic

def _hash_pt(lines):
  _dic = dict()
  for line in tqdm(lines):
    values = line.split()
    try:
      word = portugueseSnow.stem(values[0])
      if word in dictionary:
        coefs = np.asarray(values[1:], dtype='float32')
        _dic[word] = coefs
        #print("Portuguese: " + word)
    except Exception as e:
      print(e)
      print("Error on line: ")
  return _dic

def mergeDictionaries(dics_lists):
  base = dics_lists[0].copy()
  for i in range(len(dics_lists) - 1):
    base.update(dics_lists[i])

  return base

with Pool(8) as p:
  dics_lists = p.map(_hash, chunks_es)
  es_dic = mergeDictionaries(dics_lists)

  with Pool(8) as p2:
    pt_dics_lists = p2.map(_hash_pt, chunks_pt)
    pt_dic = mergeDictionaries(pt_dics_lists)
    es_dic.update(pt_dic)
    f = open("./embeddings/on-dic-fast.pkl","wb")
    pickle.dump(es_dic,f)
    f.close()

