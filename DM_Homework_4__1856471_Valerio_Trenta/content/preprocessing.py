#preprocessing.py
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
from os.path import dirname
curr = os.getcwd()
directory = dirname(curr)
print(directory)
with open(directory+"/content/lyrics.csv") as f:
  df = pd.read_csv(f, names=['index', 'song', 'year', 'artist', 'genre', 'lyrics'], dtype={'lyrics': str})
  #some preprocessing steps on lyrics: lower their case, and remove useless words/symbols which recur quite often
  df['lyrics'] = df['lyrics'].str.lower()
  df['lyrics'] = df['lyrics'].str.strip('[]')
  df['lyrics'] = df['lyrics'].str.strip('()')
  df["lyrics"] = df['lyrics'].str.replace('chorus','')
  df["lyrics"] = df['lyrics'].str.replace('verse','')
  df["lyrics"] = df['lyrics'].str.replace('intro','')
  lyrics = df['lyrics'].to_dict()
  genres = df['genre'].to_dict()
            
#now we only consider 'valid' lyrics, that is to say, only those lyrics that have actually
#more than 10 words and are NOT empty (there are empty lyrics in the dataset)
valid_lyrics = {}
valid_genres = {}
for key in lyrics:
    if len(str(lyrics[key]))>10:
        valid_lyrics[key] = lyrics[key]
        valid_genres[key] = genres[key]

print("\nRemaining valid lyrics and corresponding genres:\n")
print(len(valid_lyrics))
print(len(valid_genres))

#now a preprocessing step to tokenize, remove punctuation and stopwords from each lyric
preprocessed_lyrics = {}
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
for key in valid_lyrics: #for each lyric, tokenize it, remove punctuation and remove stopwords
    tokenized_lyric = tokenizer.tokenize(str(valid_lyrics[key]))
    preprocessed_lyrics[key] = []
    for word in tokenized_lyric:
        if word not in stop_words:
            if word not in preprocessed_lyrics[key]:
                preprocessed_lyrics[key].append(word)
#we now have for each lyric, a bag of words for the lyric

#these are all the genres we have
genres_final = {}
for g in valid_genres:
    if valid_genres[g] not in genres_final:
        genres_final[valid_genres[g]] = [valid_genres[g]]

print("\nClasses in original dataset:\n")
for key in genres_final:
  print(key)

#we will not consider genres such as "Other" or "Not available", we consider them
#as unclassified songs
#we only deal with these 5 classes
genre2lyrics = {}
genre2lyrics["Rock"] = []
genre2lyrics["Hip-Hop"] = []
genre2lyrics["Electronic"] = []
genre2lyrics["Jazz"] = []
genre2lyrics["Country"] = []
for key in preprocessed_lyrics:
    if valid_genres[key] in genre2lyrics:
        genre2lyrics[valid_genres[key]].append(preprocessed_lyrics[key])
print("\nClasses in our dataset:\n")
for key in genre2lyrics:
    print(key, len(genre2lyrics[key]))

#notice that we have far more lyrics/songs for Rock class and Hip Hop class with respect
#to the other classes, so we need to balance the dataset and we can do this by
#selecting a certain number of songs from each class, same number for each of them

#to make our dataset more balanced, we arbitrarily select
#only 1500 lyrics from each of the five genres, and work with them.
#if you wish to work with a larger dataset to have better results,
#set the value from 1500 to a higher 5000, 6000 or 7000 maximum - and the dataset is still balanced with
#these values.
genre2lyrics["Rock"] = genre2lyrics["Rock"][:1500] #also tried with [:5000]
genre2lyrics["Hip-Hop"] = genre2lyrics["Hip-Hop"][:1500] #also tried with [:5000]
genre2lyrics["Electronic"] = genre2lyrics["Electronic"][:1500] #also tried with [:5000]
genre2lyrics["Jazz"] = genre2lyrics["Jazz"][:1500] #also tried with [:5000]
genre2lyrics["Country"] = genre2lyrics["Country"][:1500] #also tried with [:5000]

import csv
with open(directory+"/content/preprocessed_lyrics.csv", 'wt') as of:
	writer = csv.writer(of, delimiter="\t")
	writer.writerow(['genre', 'lyric'])
	for genre in genre2lyrics:
		for lyric in genre2lyrics[genre]:
			writer.writerow([genre, lyric])

print("\nDone.\n")