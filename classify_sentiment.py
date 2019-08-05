#!/usr/bin/env python3
# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import sys
from tqdm import tqdm
from bs4 import BeautifulSoup
tqdm.pandas(dynamic_ncols=True)
import pickle
pos = open("positive_emoji_v2.csv").read().split(",")
neg = open("negative_emoji_v2.csv").read().split(",")

with open('sentiment_classifier_model.pickle', 'rb') as f:
    pkl = pickle.load(f)
    model = pkl["model"]
    vocab = pkl["vocab"]

def classify(html):
    feel = sum([e in html for e in pos]) - sum([e in html for e in neg])
    if feel > 0:
        return "pos"
    elif feel < 0:
        return "neg"
    soup = BeautifulSoup(html, "lxml")
    s = []
    for child in soup.find("p").children:
        if child.name == None:
            s.append(child)
        elif child.name == "img":
            s.append(child["alt"])
        else:
            s.append(child.text)
    words = " ".join(s).lower().split()
    features = {word: word in words for word in vocab}
    if not any(features.values()):
        return "?"
    return model.classify(features)

os.makedirs("classified", exist_ok=True)
files = sys.argv[1:]
#files = sorted(glob.glob("egypt tweets/egypt_tweets*.csv"))
for f in files:
    print("Loading " + f)
    df = pd.read_csv(f, sep=";")
    df["feel"] = df.html.progress_apply(classify)
    new_filename = "classified/" + os.path.splitext(os.path.basename(f))[0] + ".csv"
    df.to_csv(new_filename, sep=";", index=False)
    print("Classification done: {} positive {} negative {} undetermined".format(sum(df.feel == "pos"), sum(df.feel == "neg"), sum(df.feel == "?")))

