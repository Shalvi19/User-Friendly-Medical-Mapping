from typing import Dict, List, Tuple

import nltk
from nltk.corpus import twitter_samples    # sample Twitter dataset from NLTK
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
import spacy
from wordcloud import WordCloud, STOPWORDS
from spacy.util import minibatch,compounding
import re

data_train=pd.read_csv('drugsComTrain_raw.csv')


# nlp0=spacy.load('en_core_web_sm')
# ner0=nlp0.get_pipe('ner')


def process_review(review):
    processed_token=[]
    for token in review.split():
        token=''.join(e.lower() for e in token if e.isalnum())
        processed_token.append(token)
    return ' '.join(processed_token)

all_drugs=data_train['drugName'].unique().tolist()
all_drugs=[x.lower() for x in all_drugs]

count=0
Train_data=[]
for _,item in data_train.iterrows():
    ent_dict={}
    if count < 1000:
        review=process_review(item['review'])

        visited_items=[]
        Drugs_Scrapped=[]
        entities=[]
        for token in review.split():
            if token in all_drugs:
                for i in re.finditer(token,review):
                    if token not in visited_items:
                        entity=(i.span()[0],i.span()[1],'DRUG')
                        Drugs_Scrapped.append(review[int(entity[0]):int(entity[1])])
                        print(Drugs_Scrapped)
                        visited_items.append(token)
                        entities.append(entity)

        if len(entities) > 0:
             ent_dict['entities']=entities
             train_item=(review,ent_dict)
             Train_data.append(train_item)
             count+=1

print(Train_data)
