import os
import re
import random
import copy
from functools import partial
from collections import Counter, OrderedDict
import unicodedata
import pandas as pd
import json
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import logging


tqdm.pandas(desc="prog_bar")
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger('__file__')

class Relations_Mapper(object):
    def __init__(self, relations):
        self.rel2idx = {}
        self.idx2rel = {}
        
        logger.info("Mapping relations to IDs...")
        self.n_classes = 0
        for relation in tqdm(relations.unique()):
            if relation not in self.rel2idx.keys():
                self.rel2idx[relation] = self.n_classes
                self.n_classes += 1
        
        for key, value in self.rel2idx.items():
            self.idx2rel[value] = key
    def get_classes(self):
        od = OrderedDict(sorted(self.idx2rel.items()))
        return list(od.values())



TYPED_ENTITIES = False
nrm_types = {'ORGANIZATION' : 'ORG',
             'ORG' : 'ORG',
             'LOCATION' : 'LOC',
             'LOC' : 'LOC',
             'PERSON' : 'PER',
             'PER' : 'PER'}

def internal_add_entity_markers(row, open_marker_fmt, close_marker_fmt):
    dct_em = Counter([ item['text'] for item in row.entityMentions ])
    dct_types = { item['text'] : nrm_types[item['label']] for item in row.entityMentions }
    row.em1Text = unidecode(row.em1Text)
    row.em2Text = unidecode(row.em2Text)
    e1_num_occur = dct_em[row.em1Text]
    e2_num_occur = dct_em[row.em2Text]
    
    
    # filter out samples that have both E1 and E2 > 1 occur - for simplicity - there are only few of them
    if e1_num_occur > 1 and e2_num_occur > 1:
        return None
    
    # Anchor is the em (1 or 2) that only have 1 occur. The other have >=1. 
    # Other occur closest to anchor will be selected. Search from anchor, before (rfind) and after and compare distances in chars
    # If one em contains the other (ex: Lake Washington contains Washington) --> it is also the anchor, to prevent find Washington within Lake Washington
    if e1_num_occur > 1 or row.em2Text.find(row.em1Text) != -1:
        anchor = row.em2Text
        anc_no = 2
        other = row.em1Text
        oth_no = 1
    else:
        anchor = row.em1Text
        anc_no = 1
        other = row.em2Text
        oth_no = 2
    
        
    row.sents = unidecode(row.sents)
    idx_anchor = row.sents.find(anchor)
    # If em text cannot be found - filter sample
    if  idx_anchor == -1:
        print(anchor + '****' + row.sents)
        return None
    idx_anchor = row.sents.index(anchor)
    idx_other_before = row.sents.rfind(other,0,idx_anchor)
    idx_other_after = row.sents.find(other,idx_anchor+len(anchor))
    if idx_other_before == -1:
        idx_other = idx_other_after
    elif idx_other_after == -1:
        idx_other = idx_other_before
    else:
        idx_other = idx_other_before if abs(idx_other_before - idx_anchor) < abs(idx_other_after - idx_anchor) else idx_other_after
    
    # If em text cannot be found - filter sample
    if  idx_other == -1:
        print(other + '****' + row.sents)
        return None
    
    sent = row.sents
    def replace_other():        
        nonlocal sent
        open_marker = open_marker_fmt.format(ent_no=oth_no,ent_type=dct_types[other])
        close_marker = close_marker_fmt.format(ent_no=oth_no,ent_type=dct_types[other])
        sent = sent[:idx_other] + f'{open_marker}{other}{close_marker}' + sent[idx_other+len(other):] 
    def replace_anchor():
        nonlocal sent
        open_marker = open_marker_fmt.format(ent_no=anc_no,ent_type=dct_types[anchor])
        close_marker = close_marker_fmt.format(ent_no=anc_no,ent_type=dct_types[anchor])
        sent = sent[:idx_anchor] + f'{open_marker}{anchor}{close_marker}' + sent[idx_anchor+len(anchor):] 
        
    # Must first add markers to the right most entity - not to push indexes
    if idx_other > idx_anchor:
        replace_other()
        replace_anchor()
    else:
        replace_anchor()
        replace_other()        
    return sent

def process_nyt_lines(lines, add_entity_markers):
    # TODO: Multi label: Select a single label or create 2 samples (same vec for 2 labels ...)
    ### TODO:Debug:Remove 
    # data_path = r'..\Datasets\New York Times Relation Extraction\train.json'    
    
    lst_dicts = [json.loads(line) for line in lines]
    df = pd.DataFrame(lst_dicts)
    df = df.explode('relationMentions')
    df = pd.concat([df,df.relationMentions.apply(pd.Series)],axis=1).drop(columns='relationMentions')
    df = df.rename(columns={'sentText' : 'sents','label' : 'relations'})    
    
    
    df['sents'] = df.apply(add_entity_markers, axis=1)
    df = df[~df.sents.isna()] # filter out samples that have both E1 and E2 > 1 occur - for simplicity - there are only few of them
        
    df = df[['sents','relations']].drop_duplicates()
    return df

DEFAULT_INC_CLASSES = ['/location/location/contains', '/people/person/place_lived', '/business/person/company']        
def filter_nyt(df,balance,include_classes = DEFAULT_INC_CLASSES, subsample_n = 0):
    
    from imblearn.under_sampling import RandomUnderSampler
    
    
    print(f"Filter nyt to 3 classes. fix imbalance={balance}. subsample to {subsample_n}\n Classes={include_classes}")
    df_res = df[df.relations.isin(include_classes)]
    if balance:
        rus = RandomUnderSampler()
        X_resampled, y_resampled = rus.fit_resample(X=df_res[['sents']], y=df_res[['relations']])
        df_res = pd.concat([X_resampled,y_resampled], axis=1)
        
    if subsample_n > 0:        
        df_res = df_res.groupby('relations').sample(n=subsample_n)
        
    return df_res

def create_nyt_train_test(args):
    """
    class D:
        pass
        
    args = D()
    args.train_data = r'../Datasets/New York Times Relation Extraction/train.json'
    args.test_data = r'../Datasets/New York Times Relation Extraction/valid.json'        
    args.add_entity_markers = partial(internal_add_entity_markers,open_marker_fmt='[E{ent_no}]',close_marker_fmt='[/E{ent_no}]')
    args.filter_nyt =  1000
    """
    data_path = args.train_data #'./data/New York Times Relation Extraction/train.json'
    logger.info("Reading training file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    
    df_train = process_nyt_lines(lines,add_entity_markers = args.add_entity_markers)    
    if args.filter_nyt > 0:
        df_train = filter_nyt(df_train, balance=True,subsample_n = args.filter_nyt)        
        
    data_path = args.test_data #'./data/New York Times Relation Extraction/valid.json'
    logger.info("Reading test file %s..." % data_path)
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
    
    df_test = process_nyt_lines(lines,add_entity_markers = args.add_entity_markers)    
    if args.filter_nyt > 0:
        df_test = filter_nyt(df_test, balance=False,subsample_n = False)        
            
    return df_train, df_test

def convert_markers_to_tokens_positions(df,rm):
     """
     Prms: df with sents (text with entity markers as below), relation: 'person-company', rm (maps relation to index)
     Convert 'Hello [__E1__LOC]She[__E1__END]code' -->  {token: ['Hello','She','codes'], subj_start : 1, subj_type:'LOC', ....  } 
     """    
     vectorizer = CountVectorizer(binary = True)
     word_tokenizer = vectorizer.build_tokenizer()    
     def convert_to_tokens(row):
          tokens = word_tokenizer(row.sents)
          res_tok = []
          d = {}
          i = 0
          for tok in tokens:
               if tok == '__E1__END':
                    d['subj_end'] = i - 1
                    continue
               if tok == '__E2__END':
                    d['obj_end'] = i - 1
                    continue
               if tok.startswith('__E1'):
                    d['subj_start'] = i
                    d['subj_type'] = tok.split('__')[-1]
                    continue
               if tok.startswith('__E2'):
                    d['obj_start'] = i
                    d['obj_type'] = tok.split('__')[-1]
                    continue
               i+=1
               res_tok+=[tok]
          d['token'] = res_tok
          d['relation'] = row.relations
          d['relation_id'] = rm.rel2idx[d['relation']]
          return d 
               
     df['data'] = df.apply(convert_to_tokens, axis=1)
     return df

def create_nyt_tokens_format(args):
     """
     Loads and formats NYT with typed-markers (inline) --> convert it to dict of tokens list + subj_start, obj_startm obj_type .... format   
     class D:
         pass
         
     args = D()
     args.train_data = r'../Datasets/New York Times Relation Extraction/train.json'
     args.test_data = r'../Datasets/New York Times Relation Extraction/valid.json'        
     Test:df_test.iloc[0].data 
          for i,tok in enumerate(df_test.iloc[0].data['token']):
               print(i, tok)
     
     """
     args.add_entity_markers = partial(internal_add_entity_markers,open_marker_fmt='[__E{ent_no}__{ent_type}]',close_marker_fmt='[__E{ent_no}__END]')
     df_train, df_test = create_nyt_train_test(args)
     
     rm = Relations_Mapper(df_train['relations'])
     df_train = convert_markers_to_tokens_positions(df_train,rm)
     df_test = convert_markers_to_tokens_positions(df_test,rm)
     return df_train, df_test
