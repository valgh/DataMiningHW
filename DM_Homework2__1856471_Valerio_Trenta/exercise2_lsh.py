#EXERICSE2_LSH
print("Welcome! First things first, we are going to import the Kijiji dataset we have created in the previous steps:\n")
import pandas as pd
import hashlib
import itertools
import time
import csv
import os
directory = os.getcwd()
with open(directory+"/dataset_kijiji_cleared.tsv") as f:
	df = pd.read_csv(f, sep='\t', names=['links', 'titles', 'timestamps', 'locations', 'descriptions', 'prices'])
	#we care about descriptions only: links are like ids for our documents, but we are going to give them
	#their own ids (integers).
	descriptions = df['descriptions'][2:].to_dict() #exclude first 2 rows since they are "little mistakes" in building dataset
	#we will need links at the end anyway, so forget what I said before
	links = df['links'][2:].to_dict() #same as descriptions

print("Done!\n")

collection = []
d=2 #"lazy" way to start indexing the same way the dictionary is indexed
for desc in list(descriptions):
	collection.append((d,descriptions[desc]))
	d+=1

#SHINGLE class: takes as input a collection (list of tuples (id, description)) and outputs a list again.
#Working with lists makes it easier to exploit the same structures here and in pyspark, even if by 
#working with dictionaries we could have gained some time, but this algorithm is going to be fast anyway.
class Shingle(object):
	def __init__(self, collection):
		self.collection = collection
		self.results = []

#the following function takes as input one document and computes its k-shingles (k=10), then it hashes every shingle, sorts the list of hashed shingles
#and outputs a tuple (id, set of hashed shingles sorted).
	def shingle(self, document):
		k = 10
		shingles = [document[1][i:i+k] for i in range(len(document[1])-k+1)]
		hashes = []
		for shingle in shingles:
			hashed_shingle = hashlib.sha1(shingle.encode('utf-8')).hexdigest()
			hashes.append(hashed_shingle)
		hashes.sort(reverse = True)
		return (document[0], hashes)
#this is just to loop over all the descriptions in the collection
	def shingle_my_documents(self):
		for document in self.collection:
			self.results.append(self.shingle(document))
		return self.results

#SIGNATURE class: input and output is the same for the previous one.
class Signature(object):
	def __init__(self, collection):
		self.collection = collection
		self.results = []
#this function is basically the same of the one which was given in the text, I'm not going to explain it.
	def hashFamily(self, i):
		resultSize = 10
		maxLen = 20
		salt = str(i).zfill(maxLen)[-maxLen:]
		def hashMember(x):
			return hashlib.sha1((str(x)+salt).encode('utf-8')).hexdigest()[-resultSize:]
		return hashMember
#This function takes one document and computes its signature. We decided to have signatures of length 100, which means
#we have set the 'resultSize' variable of the function above to be eqaul to 10, and then for each set of hashed shingles we have for the 
#description, we compute the hashes of each shingle for 10 times and then, each time, we take the minimum among the hashes computed for every shingle,
#so in the end 'min_sign' will be composed of 10 hashed values of length 10.
	def compute_min_hashing_signature(self, document):
		min_sign = ""
		print("compute_min hash signature for document %d..." %document[0])
		for i in range(0,10):
			hashes = []
			hF = self.hashFamily(i)
			for shin in document[1]:
				hashes.append(hF(shin))
			min_sign+=min(x for x in hashes)
		return (document[0], min_sign)
#this is just to iterate over all the shingles sets in the collection
	def signatures(self):
		for document in self.collection:
			self.results.append(self.compute_min_hashing_signature(document))
		return self.results

#LSH CLASS:
#each document's signature has length 100, so we will choose for this n_bands=10 and r(rows)=10
#threshold t = (1/n_bands)^(1/r) = 0.7943 = 0.80 approx. We expect to have probability equal to 70% (more or less, see the 
#S-curve plotted with WOlframAlhpa in 'image_results') to mark each pair of descriptions which exceeds the threshold as "near-duplicates",
#meaning that we still expect to have some false negatives and, obviously, false positives with this method.
class Lsh(object):
	def __init__(self, collection):
		self.collection = collection
		self.results = []
#this function makes sense if you look it together with the one below it. It basically takes a signature, the number of band desired
#(if the first one, the second or so on...) and returns the row of the signature falling under that band.
	def get_band(self, signature, band):
		n_bands = 10
		start = ((100/n_bands)*band)-10
		end = (100/n_bands)*band
		return signature[int(start):int(end)]
#this is where buckets are created. This function takes as input one single signature (document), so it considers its id and the signature itself
#since from the previous function document = (id, signature). Then, in the for-loop it gets the current band with the previous function
#where i = current band, and appends to the output the tuple ((i, bucket), doc_id) where i = band, bucket = hash of the bucket to which the
#row at band i of the signature hashes to, and doc_id is again the integer id of the description.
	def buckets(self, document):
		doc_id, sign = document
		buckets = []
		for i in range(1,11):
			current_band = self.get_band(sign, i)
			buckets.append(((i, hashlib.sha1(current_band.encode('utf-8')).hexdigest()), doc_id))
		return buckets
#this function makes sense if you look it together with the LSH function. What I do here is, for a bucket in input,
#get the combinations of the descriptions that hashes to that bucket and classify all of them as candidate pairs
	def candidate_pairs(self, bucket):
		candidate_pairs = []
		list_of_pairs = list(itertools.combinations(bucket,2))
		for (doc_A, doc_B) in list_of_pairs:
			candidate_pairs.append(((doc_A, doc_B), 1))
		#if candidate_pairs:
		return candidate_pairs
#for each candidate, now, we compare its score against the threshold and, if >=0.8, append it to the results
#marking it as a duplicate pair.
	def compute_results(self, candidate):
		results = []
		n_bands = 10
		t = 0.8
		for pair in candidate:
			score = candidate[pair]
			if score/n_bands>=t:
				results.append((pair, score/n_bands))
		return results
#this is where all the logic is implemented, 'results' is our output.
	def lsh(self):
		print("\nStarting LSH phase...\n")
		buckets = [] #structure to hold all the buckets (band, hash)
		dict0 = {}
		candidatePairs = []
		dict1 = {}
		for document in self.collection:
			buckets.append(self.buckets(document)) #for every description, compute the buckets
		for bucket in buckets: #for every bucket you have
			for h in bucket:
				group, doc = h #split the bucket tuple ino (band, hash) and doc_id
				if group not in dict0: #in dict0 we will have '{(band, hash) : [list of descriptions in bucket]}'
					dict0[group] = [doc]
				else:
					dict0[group].append(doc)
		for group in dict0: #for every bucket in dict0
			candidatePairs.append(self.candidate_pairs(dict0[group])) #compute the candidate pairs and append it to a list
		for li in candidatePairs: #candidatePairs is [[((doc_A, doc_B), 1)]]
			if li:
				for candidate in li:
					pair, score = candidate
					if pair not in dict1:
						dict1[pair] = score #in dict1 we have '{(doc_A, doc_B) : score}'. 'score' is #of bands the two documents hash to same bucket
					else:
						dict1[pair]+=score #if you meet (doc_A, doc_B) more than once, +1 to the score
		self.results = self.compute_results(dict1)
		return self.results

#starting LSH...
print("\nLSH PHASE BEGINNING:\n")
start = time.time()
s = Shingle(collection)
h = Signature(s.shingle_my_documents())
l = Lsh(h.signatures())
results = l.lsh()
end = time.time()
#now need to comprehend the situation here. For each document, we attach to it its near-duplicates.
dup_of = {}
for dup in results:
	pair,score = dup
	if pair[0] not in dup_of:
		dup_of[pair[0]] = [pair[1]]
	else:
		dup_of[pair[0]].append(pair[1])
#and now we can enumerate the duplicates.Since we have combinations, here we are basically stating that for each "group" of
#duplicates, the only one which we do not consider duplicate and will never append to the list is the first one being 
#enumerated in the combinations. The other ones will all be appended here.
#Doesn't scale well, but should be ok for 2300 descriptions.
enum = []
for dup in dup_of:
	for d in dup_of[dup]:
		if d not in enum:
			enum.append(d)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++TESTING+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#this is going to be used for testing...had to place it here for memory reasons on my pc, if put after LSH execution memory allocated won't make it
#finish in a decent time, not even cleaning the garbage collector helped so had to place it here.
print("\nPREPARING FOR TESTING...\n")
print("\nComputing the duplicates with simple comparison, this could take a while (took approx. 6/7 minutes on my pc)...\n")
start_2 = time.time()
s = Shingle(collection)
shins = s.shingle_my_documents()
list_of_pairs = list(itertools.combinations(shins,2)) #had to use list here anyway since set or dictionaries wouldn't work with itertools
#clearing the dataset helped in making the running time slightly better anyway.
output = {}
for doc_A, doc_B in list_of_pairs:
	cnt = (len(set(doc_A[1]).intersection(set(doc_B[1]))))/(len(set(doc_A[1]).union(set(doc_B[1]))))
	if cnt >= 0.8:
		output[(doc_A[0], doc_B[0])] = cnt
end_2=time.time()

#get the number of duplicates the same way we selected them for the LSH method
dup_of_comparison = {}
for pair in output:
	if pair[0] not in dup_of_comparison:
		dup_of_comparison[pair[0]] = [pair[1]]
	else:
		dup_of_comparison[pair[0]].append(pair[1])
enum_comparison = []
for dup in dup_of_comparison:
	for d in dup_of_comparison[dup]:
		if d not in enum_comparison: #this doesn't scale well, as stated before, but it's good enough for 2300 descriptions I think
			enum_comparison.append(d)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++END TESTING+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#computing the intersection, nothing to explain here
print("\nDone! Now computing the intersection between the methods to test, wait...\n")
intersection = 0
for doc in enum_comparison:
	if doc in enum:
		intersection+=1

#printing the results...

print("\nNEAR-DUPLICATE PAIRS FOUND:\n")
print(results)
print("\nNUMBER OF NEAR-DUPLICATE PAIRS FOUND WITH LSH:\n")
print(len(results))
print("\nNUMBER OF DUPLICATES FOUND WITH LSH:\n")
print(len(enum))
print("\nTIME OF THE EXECUTION (LSH):\n")
print(end-start)
print("\nNUMBER OF NEAR-DUPLICATE PAIRS FOUND WITH COMPARISON OVER SHINGLES SETS:\n") #THIS IS THE REQUESTED ONE
print(len(enum_comparison))
print("\nTIME OF THE EXECUTION FOR THE PREVIOUS TASK:\n") #THIS IS THE REQUESTED ONE
print(end_2-start_2)
print("\nIntersection size between lsh and this method:\n") #THIS IS THE REQUESTED ONE
print(intersection)

#write results into file
print("\nWRITING THE RESULTS INTO FILE...\n")
with open(directory+"/duplicates_lsh.tsv", "wt") as of: #write the obtained data into a .tsv file
	writer = csv.writer(of, delimiter="\t")
	writer.writerow(['id', 'link'])
	for d in enum:
		writer.writerow([d, links[d]])

print("\nDone. You can find the links of the duplicates under the file 'duplicates_lsh.tsv'.\n")

#So the results are the one we expected: the intersection between the two methods is large enough to state that
#LSH algorithm works, and actually finds almost every near-duplicate with a very small error - due to the fact we have only 70% 
#of probability to mark as near-duplicates a pair of near-duplicate descriptions for the threshold and number of bands/rows
#we chose. So of course we expect some false negatives and false positives as stated before, but the running time for the LSH algorithm
#is a huge advantage considering the one of the lazy and greedy algorithm exploited for the shingles sets.
#A screenshot of the obtained results can be found under the 'images_results' folder.
