from datetime import timedelta, datetime
import glob
from itertools import chain
import json

import os
import re
import pdb

import fire
from tqdm import tqdm

import numpy as np
import pandas as pd

from pandas.io.json import json_normalize
from collections import Counter
import scipy.sparse as spr
import pickle
import implicit

from arena_util import write_json


# open & load file

with open('./res/val.json') as f:
    v = json.load(f)
with open('./res/train.json') as f:
    t = json.load(f)

val_rawdata = json_normalize(v)
train_rawdata = json_normalize(t)

val_map = val_rawdata[['id','songs','tags','updt_date']]
val_map["is_train"] = 0

train_map = train_rawdata[['id','songs','tags','updt_date']]
train_map["is_train"] = 1

# train data num & test data num
n_train = len(train_map)
n_test = len(val_map)

# train + test
plylst = pd.concat([train_map, val_map], ignore_index=True)

# total playlist id = nid  
plylst["nid"] = range(n_train + n_test)

# id <-> nid mapping
plylst_id_nid = dict(zip(plylst["id"],plylst["nid"]))
plylst_nid_id = dict(zip(plylst["nid"],plylst["id"]))

# song id <-> sid mapping 
plylst_song = plylst['songs']
song_counter = Counter([sg for sgs in plylst_song for sg in sgs])
song_dict = {x: song_counter[x] for x in song_counter}

song_id_sid = dict()
song_sid_id = dict()
for i, t in enumerate(song_dict):
  song_id_sid[t] = i
  song_sid_id[i] = t

n_songs = len(song_dict)

# tag id <-> tid mapping 
plylst_tag = plylst['tags']
tag_counter = Counter([tg for tgs in plylst_tag for tg in tgs])
tag_dict = {x: tag_counter[x] for x in tag_counter}

tag_id_tid = dict()
tag_tid_id = dict()
for i, t in enumerate(tag_dict):
  tag_id_tid[t] = i
  tag_tid_id[i] = t

n_tags = len(tag_dict)

# add sid & tid to plylst table
plylst['songs_id'] = plylst['songs'].map(lambda x: [song_id_sid.get(s) for s in x if song_id_sid.get(s) != None])
plylst['tags_id'] = plylst['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])

# new plylst table for learning
plylst_use = plylst[['is_train','nid','updt_date','songs_id','tags_id']]
plylst_use.loc[:,'num_songs'] = plylst_use['songs_id'].map(len)
plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)
plylst_use = plylst_use.set_index('nid')
n_plylsts = len(plylst_use)

#
plylst_train = plylst_use.iloc[:n_train,:]
plylst_test = plylst_use.iloc[n_train:,:]

# real test
test = plylst_test

# make csr_matrix
row = np.repeat(range(n_plylsts), plylst_use['num_songs']) 
col = [song for songs in plylst_use['songs_id'] for song in songs]
dat = np.repeat(1, plylst_use['num_songs'].sum())
train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_plylsts, n_songs))

row = np.repeat(range(n_plylsts), plylst_use['num_tags'])
col = [tag for tags in plylst_use['tags_id'] for tag in tags]
dat = np.repeat(1, plylst_use['num_tags'].sum())
train_tags_A = spr.csr_matrix((dat, (row, col)), shape=(n_plylsts, n_tags))

# transpose
train_songs_A_T = train_songs_A.T.tocsr()
train_tags_A_T = train_tags_A.T.tocsr()

# Song 추천을 위한 Model 초기화
song_recommend_model = implicit.als.AlternatingLeastSquares(factors=20,  regularization=0.1, iterations=100)

# Calculate the confidence by multiplying it by our alpha value.
alpha_val = 15
song_conf = (train_songs_A_T * alpha_val).astype('double')

# Model 학습
song_recommend_model.fit(song_conf)

# Tag 추천을 위한 Model 초기화
tag_recommend_model = implicit.als.AlternatingLeastSquares(factors=20,  regularization=0.1, iterations=100)

# Calculate the confidence by multiplying it by our alpha value.
tag_conf = (train_tags_A_T * alpha_val).astype('double')

# Model 학습
tag_recommend_model.fit(tag_conf)

answers = []

for nid in tqdm(test.index):

    recommendations_songs_tuples = song_recommend_model.recommend(int(nid), train_songs_A, 100)
    recommendations_tags_tuples = tag_recommend_model.recommend(int(nid), train_tags_A, 10)

    # extract only songs/tags from (songs/tags, score) tuple
    recommendations_songs = [t[0] for t in recommendations_songs_tuples]
    recommendations_tags = [t[0] for t in recommendations_tags_tuples]

    ans_songs = [song_sid_id[song] for song in recommendations_songs]
    ans_tags = [tag_tid_id[tag] for tag in recommendations_tags]

    answers.append({
                "id": plylst_nid_id[nid],
                "songs": ans_songs,
                "tags": ans_tags
            })
    
# write_json 
write_json(answers, "results/results.json")
