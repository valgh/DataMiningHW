#KIJIJI SEARCH ENGINE: PREPROCESSING
#we assume the datase is obviously given in the format we have already exploited in the previous homework, but 
#preprocessed so that the duplicate links do not appear.
import pandas as pd
import os
#we exploit pandas since it lets us select only those columns of the file we are interested in!
directory = os.getcwd() 
with open(directory+"/dataset_kijiji_cleared.tsv") as f:
	df = pd.read_csv(f, sep='\t', names=['links', 'titles', 'timestamps', 'locations', 'descriptions', 'prices'])
	links = df['links'].to_dict()
	titles = df['titles'].to_dict()
	timestamps = df['timestamps'].to_dict()
	locations = df['locations'].to_dict()
	descriptions = df['descriptions'].to_dict()
	prices = df['prices'].to_dict()

#now we begin with the preprocessing phase:
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer 

#preprocessing is applied in sequence: for every word in the description,
#turni it to lower case, then tokenize it, then for every token check if it's not a stopword and,
#if it's not, take it and stem it.

def to_lower_case(dictionary):
	output = {}
	for k in dictionary:
		output[k] = dictionary[k].lower() #nothing much to say, just transforms every description to lower case
	return output

def tokenize(dictionary):
	tokens = {}
	tokenizer = RegexpTokenizer(r'\w+') #Regexp also filters out punctuation, so we have only tokens of words
	for k in dictionary:
		tokens[k] = tokenizer.tokenize(dictionary[k])
	return tokens


def stopwords_removal(tokens):
	swords = set(stopwords.words('italian')) #the italian stopwords are not many
	filtered_tokens = {} #this will be the output
	for k in tokens:
		for w in tokens[k]:
			if w not in swords: #if token not in stopwords set
				if k not in filtered_tokens:
					filtered_tokens[k] = [w] #put it in output
				else:
					filtered_tokens[k].append(w)
	return filtered_tokens

def stemmer(filtered_tokens):
	s = SnowballStemmer('italian') #SnowballStemmer works fine with Italian language
	final_output = {}
	for k in filtered_tokens:
		for w in filtered_tokens[k]:
			if k not in final_output:
				final_output[k] = [s.stem(w)]
			else:
				final_output[k].append(s.stem(w))
	return final_output


print("\nTITLES WITH PREPROCESSING:\n")
#in sequence, as stated before
final_titles = stemmer((stopwords_removal(tokenize(to_lower_case(titles)))))
print(final_titles)
print("\nDESCS WITH PREPROCESSING:\n")
#and same for descriptions
final_descriptions = stemmer((stopwords_removal(tokenize(to_lower_case(descriptions)))))
print(final_descriptions)
#Join results in a .tsv file, we exploit what we have already done in the previous homework
import csv
with open(directory+"/preprocessed_dataset_kijiji.tsv", 'wt') as of:
	writer = csv.writer(of, delimiter="\t")
	writer.writerow(['links', 'titles', 'timestamps', 'locations', 'descriptions', 'prices'])
	for i in links: 
		writer.writerow([links[i], ' '.join(final_titles[i]), timestamps[i], locations[i], ' '.join(final_descriptions[i]), prices[i]])

print("\nDone. Preprocesed data available as 'preprocessed_dataset_kijiji.tsv'.\n")

