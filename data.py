# Secret:qY8nY9ehi_cG8CCgJKW5tHrTI0MuKA
# ID: 0D31RjG_Ajl7VxOg6Hh5dA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import praw
import nltk

user_agent = "Scraper 1.0 by /by/PinPractical5955"
reddit = praw.Reddit(
    client_id="0D31RjG_Ajl7VxOg6Hh5dA",
    client_secret="qY8nY9ehi_cG8CCgJKW5tHrTI0MuKA",
    user_agent=user_agent
)

# can use hot, new, rising, or top
headlines = []
for submission in reddit.subreddit('wallstreetbets').hot(limit=None):
    data = {
        "Submission" : submission,
        'Title': submission.title,
        'Author': submission.author,
        'Number of comments': submission.num_comments,
        'Date of Post': submission.created_utc,
        "Number of Upvotes" : submission.upvote_ratio,
        "Tag" : submission.author_flair_text,
    }
    headlines.append(data)
print(len(headlines))

df = pd.DataFrame(headlines)
print(df.head())

df.to_csv('wallstreetbets.csv', header=False, encoding='utf-8', index=False)