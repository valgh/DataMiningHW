#KIJIJI SEARCH ENGINE: IMPLEMENTATION OF THE INDEX
#we load the .tsv preprocessed dataset we have created in the previous step of this exercise
import csv
import math
import os
inverted_index={}
bag_of_words = []
counting_links={}
counting_terms={}
count_terms_links = {}
directory = os.getcwd()
with open(directory+"/preprocessed_dataset_kijiji.tsv") as f: #just to count the number of documents in the dataset. We already know 
#how many documents we have, but if the number changes, we need to count it anyway
	reader = csv.reader(f, delimiter = "\t")
	#for row in reader:
	print("\nCounting number of announcements...\n") #a double check from when I didn't adjust the original dataset. can be ignored.
	for i, row in enumerate(reader):
		if i>2: #leave out first rows that are not useful
			link = row[0]
		#to compute the tf-idf score.
			if link not in counting_links:
				counting_links[link]=link
print("\nNumber of announcements we have (N): \n") 
print(len(counting_links))

with open(directory+"/preprocessed_dataset_kijiji.tsv") as f:
	#we now need to count the number of links each term appears in. We need this to compute the tf-idf score correctly.
	reader = csv.reader(f, delimiter = "\t")
	for r, row in enumerate(reader):
		if r>2: #leave out first rows that are not useful
			#print("Processing row: ", r)
			link = row[0]
			title = row[1]
			description = row[4]
			bag_of_words = description.split(" ")+title.split(" ") #the bag of words is 'renewed' for each row
			counting_terms[link] = len(bag_of_words)
			for term in bag_of_words:
				if term in count_terms_links:
					if link not in count_terms_links[term]:
						count_terms_links[term][link] = 1
					else:
						count_terms_links[term][link]+=1
				else:
					count_terms_links[term] = {link : 1}
#print(count_terms_links)
#print(counting_terms)

#Now we have everything we need to compute the tdf-idf score for each description!
	#what we do now is: for each announcement, take terms from title and description (already done in the previous step basically)
	#put each term into an inverted index such as term -> (announcement Link(docID), tf-idf score)
	#this is where we compute the tf-idf score:
	#tf-idf score is obtained as: tf-idf = [tf](log(#of occurrences of term in link))*
	#*[idf](log(#of documents in collection/1+#of documents containing the term)))
	#where that '1+' is a smoothing parameter not to have errors like a division by 0.

for term in count_terms_links:
	for link in count_terms_links[term]:
		tf = ((count_terms_links[term][link]))#/(counting_terms[link]))
		if tf>0:
			tf = 1+(math.log10(tf))
		idf = 1+(math.log10((len(counting_links))/(len(count_terms_links[term]))))
		if term not in inverted_index:
			inverted_index[term] = {link : tf*idf}
		else:
			inverted_index[term][link] = tf*idf 

print("\nThis is the obtained inverted index:\n")
print(inverted_index)
#writing the obtained inverted index to file
with open(directory+"/inverted_index_dictionary_kijiji.tsv", "wt") as of: #write the obtained data into a .tsv file
	writer = csv.writer(of, delimiter="\t")
	writer.writerow(['term', 'link'])
	for term in list(inverted_index):
		writer.writerow([term, inverted_index[term]])

#the tf-idf scores we obtain are all greater than 0 and in some cases even than one, which is still ok: the important thing is that they are NOT 
#NEGATIVE.
#seeing the results we obtain, they should be good.

print("\nDone. The inverted index can be found as 'inverted_index_dictionary_kijiji.tsv'.\n")