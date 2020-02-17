#QUERY_MY_INDEX
import time
import os
read = input("\nEnter your query (press 'h' for standard query 'bilocale arredato'): ")
start = time.time()
if read == 'h':
	query = "bilocale arredato"
else:
	query = read
inverted_index = {}
import pandas as pd
directory = os.getcwd()
with open(directory+"/inverted_index_dictionary_kijiji.tsv") as f:
	reader = pd.read_csv(f, sep='\t', names=['terms', 'links'])
	terms = reader['terms'].to_dict()
	links = reader['links'].to_dict()

#preprocessing the query. We need to preprocess it the exact same way we preprocessed our descriptions in dataset.
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer

def to_lower_case(query):
	return query.lower()

def tokenize(query):
	tokens = {}
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(query)
	return tokens

def stopwords_removal(tokens):
	swords = set(stopwords.words('italian'))
	filtered_tokens = []
	for word in tokens:
		if word not in swords:
			filtered_tokens.append(word)
	return filtered_tokens

def stemmer(filtered_tokens):
	s = SnowballStemmer('italian')
	final_output = {}
	for k in filtered_tokens:
		if k not in final_output:
			final_output[k] = s.stem(k)
	return list(final_output.values())


print("Query preprocessed:\n")
query_preprocessed = stemmer((stopwords_removal(tokenize(to_lower_case(query)))))
print(query_preprocessed)
import re
import math
import pandas as pd
#we are going to need this for the normalization phase:
with open(directory+"/preprocessed_dataset_kijiji.tsv") as f:
	reader = pd.read_csv(f, sep='\t', names=['links', 'titles', 'timestamps', 'locations', 'descriptions', 'prices'])
	lks = reader['links'].to_dict()
	titles = reader['titles'].to_dict()
	descriptions = reader['descriptions'].to_dict()

#now we need to compute the cosine score for the query_preprocessed we obtained.
#the algorithm was taken from IR Book, chapter 6, so if you need to read a pseudocode, it's there, around page 125.
#had to exploit the re module to parse the list of links-tfidf scores contained in each term of the ivnerted index,
#I think it worked fine.

def cosine_score(query_preprocessed):
	fkey = 0
	separator1 = ","
	separator2 = "':" #"constants" to parse the inverted index entry
	scores={} #this is going to be the output
	query_tfidf={} #this is the value of the tfidf score for the query
	postings_lists={} #these are the postings lists to retrieve the results form inverted index
	n_docs = {} #this is going to be used to count the number of documents in which a term occurs
	for term in query_preprocessed: #this is where we compute the tf-idf score for the query
		for key in terms:
			if terms[key]==term:
				postings_lists[term] = links[key]
				split_to_count = re.split(separator1, postings_lists[term])
				split_to_count[-1] = split_to_count[-1][:-1]
				n_docs[term] = len(split_to_count)
		query_tfidf[term] = (((1/(len(query_preprocessed)))))*(1+(math.log10((len(lks)-2)/(n_docs[term])))) #tf-idf score for the query
		#by computing the tf-idf score for the query like this, we have the first term (tf) for the query which depends on the
		#query itself, and the second term (idf) which depends on the corpus of our dataset, that is the postings lists retrieved
		#for the terms in the query
	for term in postings_lists: #this is where we begin to compute the cosine score
		split_list = re.split(separator1, postings_lists[term])
		split_list[-1] = split_list[-1][:-1] #links
		for element in split_list:
			split_again = re.split(separator2, element) #one link-score
			split_again[0] = split_again[0][2:] #link, split_again[1] is going to be the score
			if split_again[0] not in scores:
				scores[split_again[0]] = (float(split_again[1]))*(query_tfidf[term])
			else:
				scores[split_again[0]] += (float(split_again[1]))*(query_tfidf[term])
	#length normalization phase implemented here - the phase for which we needed also the preprocessed dataset:
	length_doc=0
	for l in scores:
		for h in lks:
			if lks[h] == l:
				length_doc = len(descriptions[h])+len(titles[h])
		scores[l] = scores[l]/(length_doc)
	return scores

#print top-22 results and running time for the query.
end = time.time()
print("\nTop 22 announcements for '"+query+"'. Results are in the form: {link : score}.\n")
final_results_sorted = sorted(cosine_score(query_preprocessed).items(), key=lambda kv: kv[1], reverse=True)
print(final_results_sorted[:22])
print("\nRUNNING TIME FOR THE QUERY:\n")
print(end-start)
#for the screenshots of the obtained results, see the 'image_results' folder. You can find some very short queries and some
#large queries, two of them corresponding to descriptions of announcements in the dataset. The search engine seems to be working fine,
#it assigns identical scores to duplicates as expected (and from here, we can see there are still lots of duplicates with different links)
#and it works both for short and long queries, since we obtain the desired results.

