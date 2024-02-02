import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import seaborn as sns

# 0.) Prepare the Data
data_path = "wallstreetbets.csv"
columns = ['Submission', 'Title', 'Number of comments', 'Date of post', 'Number of Upvotes', 'Tag']
df = pd.read_csv(data_path, names=columns, header=None)
df["Id"] = None

#print(df.shape) --> (890, 6)
example = df["Submission"][50]
# print(example)
tokens = nltk.word_tokenize(example) # splitting up sentence into individual words

tagged = nltk.pos_tag(tokens) # part of speech tag
entities = nltk.chunk.ne_chunk(tagged) # chunks it into similar words/groups
# print(entities.pprint())

# VADER (Valence Aware Dictionary and Sentiment Reasoner) - Bag of words approach --> representation of text that is based on an unordered collection of words
sia = SentimentIntensityAnalyzer()
# print(sia.polarity_scores('I am depressed')) --> gives a score based on neutral, pos, or neg
#print(sia.polarity_scores(example))

# Run polarity score on entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df["Submission"])):
    text = row['Submission']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)
    # break

vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')
# print(vaders.head())



# Learn how to do sentiment analysis
# # 1.) Model
# input_size = ...
# output_size = 1
# Sentiment analysis

# model = ...

# # 2.) Loss and Optimizer
# criterion = torch.nn.CrossEntropyLoss() # trained
# learning_rate = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# # 3.) Training loop

# # 4.) Testing and Evaluation