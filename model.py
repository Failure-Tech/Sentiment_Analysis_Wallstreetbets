import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 0.) Prepare the Data
data_path = "./ML/Projects/Tesla_Predictor_Comparison/wallstreetbets.csv"
df_data = pd.read_csv(data_path)

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