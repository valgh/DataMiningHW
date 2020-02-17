#exercise3_lsh_spark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import pandas as pd
import hashlib
import itertools
import time
import csv
import os
#define the SparkContext, which we will use to parallelize the execution
sc = SparkContext("local", "LSH_PART_1", pyFiles=['exercise3_lsh_spark.py'])
print("Loading the dataset...\n")
#define SparkSession, which is going to be exploited to create the DataFrame
spark = SparkSession.builder.master("local").appName("LSH_1").getOrCreate()
directory = os.getcwd() 
with open(directory+"/dataset_kijiji_cleared.tsv") as f:
	df = pd.read_csv(f, sep='\t', names=['links', 'titles', 'timestamps', 'locations', 'descriptions', 'prices'])
	#we care about links and descriptions only: links are like ids for our documents, descriptions are what we will
	#consider to check whether two documents are duplicates or not.
	descriptions = df['descriptions'][2:].to_dict()
	#we will need links at the end anyway
	links = df['links'][2:].to_dict()
	li = []
	d=2 #just like ex2, start indexing so that it is sync with the dataset
	for desc in list(descriptions):
		li.append((d, descriptions[desc]))
		d+=1
	#created a dataframe from the list of (id, description) descriptions
	data = spark.createDataFrame(li, schema=None, samplingRatio=None, verifySchema=True)
#create the RDD and collect it
dataRDD = data.rdd.collect()
#print(dataRDD)

#WHAT WE NEED TO SHINGLE
#the same function we exploited in exercise 2
def shingle(document):
	k = 10
	shingles = [document[1][i:i+k] for i in range(len(document[1])-k+1)]
	hashes = []
	for shingle in shingles:
		hashed_shingle = hashlib.sha1(shingle.encode('utf-8')).hexdigest()
		hashes.append(hashed_shingle)
	hashes.sort(reverse = True)
	return (document[0], hashes)

#WHAT WE NEED TO MIN_HASH_SIGNATURES
#again, same functions we exploited in exercise 2, not going to comment them again
def hashFamily(i):
	resultSize = 10
	maxLen = 20
	salt = str(i).zfill(maxLen)[-maxLen:]
	def hashMember(x):
		return hashlib.sha1((str(x)+salt).encode('utf-8')).hexdigest()[-resultSize:]
	return hashMember

def compute_min_hashing_signatures(document):
	min_sign = ""
	print("compute_min hash signature for document...")
	for i in range(0,10):
		hashes = []
		hF = hashFamily(i)
		for shin in document[1]:
			hashes.append(hF(shin))
		min_sign+=min(x for x in hashes)
	return (document[0], min_sign)

#WHAT WE NEED TO LSH
#same functions from exercise 2 again, only we don't have the lsh function since we won't need it anymore -> see command below
def get_band(signature, band):
	n_bands = 10
	start = ((100/n_bands)*band)-10
	end = (100/n_bands)*band
	return signature[int(start):int(end)]

def buckets(document):
	doc_id, sign = document
	buckets = []
	for i in range(1,11):
		current_band = get_band(sign, i)
		buckets.append(((i, hashlib.sha1(current_band.encode('utf-8')).hexdigest()), doc_id))
	return buckets

def candidate_pairs(bucket):
	candidate_pairs = []
	list_of_pairs = list(itertools.combinations(bucket[1],2))
	for (doc_A, doc_B) in list_of_pairs:
		candidate_pairs.append(((doc_A, doc_B), 1))
	return candidate_pairs

def compute_results(candidate):
	results = []
	n_bands = 10
	t = 0.8
	pair, score = candidate
	if score/n_bands>=t:
		results.append((pair, score/n_bands))
	return results



start = time.time()
print("Computing with shingles of length 10 and signatures of length 100...")
#we compute the results by exploiting the Spark framework functionalities. When we exploit map, we do it because we know the output
#we will get from the function being executed by map is of the same size of the input, while 'flatMap' is exploited when the output
#size changes from the input, so of course is exploited in computing buckets, candidate pairs and final results.
#'groupByKey' output (meaning, the value) has to be mapped to a list again otherwise we can't compute the candidate pairs, but it's very useful since
#it lets us group the buckets with same (band, hash) to one single entry and, by 'mapValues(list)', we make it so that the set of descriptions ids
#hashing to the same bucket is actually returned a list.
#The 'reduceByKey()' function takes a lambda as input so that for each candidate having same key (meaning, same (doc_A, doc_B)), it sums the values
#of their score (always 1, so it's basically like implementing a count function here).
#We then compute the results by comparing the scores against the threshold, then collect the results.
collection = sc.parallelize(dataRDD).map(shingle).map(compute_min_hashing_signatures).flatMap(buckets).groupByKey().mapValues(list).flatMap(candidate_pairs).reduceByKey(lambda x,y: x+y).flatMap(compute_results).collect()
#to simulate a number of 12 machines in the process - obviously doesn't scale well, the more machines to simulate the 
#more it will take to compute the near duplicates:
#collection = sc.parallelize(dataRDD, 12).map(shingle).map(compute_min_hashing_signatures).flatMap(buckets).groupByKey().mapValues(list).flatMap(candidate_pairs).reduceByKey(lambda x,y: x+y).flatMap(compute_results).collect()
print("\nDUPLICATES FOUND:\n")
print(collection)
end = time.time()
dup_of = {}
for dup in collection:
	pair,score = dup
	if pair[0] not in dup_of:
		dup_of[pair[0]] = [pair[1]]
	else:
		dup_of[pair[0]].append(pair[1])
enum = []
for dup in dup_of:
	for d in dup_of[dup]:
		if d not in enum:
			enum.append(d)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++TESTING+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#this is going to be used for testing, same algorithm exploited on exercise 2!
#For the testing algorithm implemented in Spark, read below.
print("\nPREPARING FOR TESTING...\n")
print("\nComputing the duplicates with simple comparison, this could take a while (took approx. 6.5 minutes on my pc)...\n")
start_2 = time.time()
shins = sc.parallelize(dataRDD).map(shingle).collect()
list_of_pairs = list(itertools.combinations(shins,2))
output = {}
for doc_A, doc_B,  in list_of_pairs:
	cnt = (len(set(doc_A[1]).intersection(set(doc_B[1]))))/(len(set(doc_A[1]).union(set(doc_B[1]))))
	if cnt >= 0.8:
		output[(doc_A[0], doc_B[0])] = cnt
end_2=time.time()

dup_of_comparison = {}
for pair in output:
	if pair[0] not in dup_of_comparison:
		dup_of_comparison[pair[0]] = [pair[1]]
	else:
		dup_of_comparison[pair[0]].append(pair[1])
enum_comparison = []
for dup in dup_of_comparison:
	for d in dup_of_comparison[dup]:
		if d not in enum_comparison: #this doesn't scale well, but it's good enough for 2300 descriptions I think
			enum_comparison.append(d)
#++++++++++++++++++++++++++++++++++++++++++++++++++++SPARK-IMPLEMENTED TESTING+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Now, this was the "official" previous way to get all the results. I had a problem by implementing this part fully in pyspark,
#because the Java heap run out of memory space (of course, I have 3 million combinations...), I heard other guys had this problem too. So 
#I run this with a reduced dataset input (considering only the first 500 descriptions in the dataset) and initialized
#10 machines to parallelize the tasks, and it seemed to work just fine,
#even if of course I found a smaller number of duplicates, but considering I have only 500 descriptions out of 2380, also the running time
#obtained was very high (200 seconds at least).
#Result image is under the usual folder, which seems to be a good result for a reduced dataset.
#I decided to write the code exploited as a comment here, even if I think it's not an efficient solution for this task - you should modify a file
#in Spark configs to make it run, and run everything simulating a number n of different 'machines' in order to avoid this memory problem, of course
#with a running time even higher than before, so I decided to stop here and consider the algorithm done before, but anyway this would be
#the implementation on Spark of the testing algorithm:

#start_2 = time.time()
#we take all the combinations of the descriptions
#def combs(li):
#	list_of_pairs = list(itertools.combinations(li[:500],2)) #notice li[:500] means we take only 500 descriptions here!
#	return list_of_pairs
#lis = combs(li)
#we create a new RDD with the combinations
#dataGreedy = spark.createDataFrame(lis, schema=None, samplingRatio=None, verifySchema=True)
#dataGreedyRDD = dataGreedy.rdd.collect()
#this function computes for every pair of documents their Jaccard Similarity, it's just like what we did in exercise 2
#def sim(pair):
#	output = []
#	doc_A = shingle(pair[0])
#	doc_B = shingle(pair[1])
#	cnt = (len(set(doc_A[1]).intersection(set(doc_B[1]))))/(len(set(doc_A[1]).union(set(doc_B[1]))))
#	if cnt >= 0.8:
#		output.append(((doc_A[0], doc_B[0]), cnt))
#	return output
#output is given by a simple flatMap for function sim
#output = sc.parallelize(dataGreedyRDD, 10).flatMap(sim).collect() #here setting up 10 machines to run this task in order not ot fall out of memory 
#end_2=time.time()
#this part is exactly the same of exercise 2
#dup_of_comparison = {}
#for pair in output:
#	if pair[0][0] not in dup_of_comparison:
#		dup_of_comparison[pair[0][0]] = [pair[1][0]]
#	else:
#		dup_of_comparison[pair[0][0]].append(pair[1][0])
#enum_comparison = []
#for dup in dup_of_comparison:
#	for d in dup_of_comparison[dup]:
#		if d not in enum_comparison: #this doesn't scale well, but it's good enough for 2300 descriptions I think
#			enum_comparison.append(d)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++END TESTING+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("\nDone! Now computing the intersection between the methods to test, wait...\n")
intersection = 0
for doc in enum_comparison:
	if doc in enum:
		intersection+=1
print("\nNEAR DUPLICATES FOUND WITH LSH (SPARK):\n")
print(collection)
print("\nNUMBER OF NEAR-DUPLICATE PAIRS FOUND WITH LSH (SPARK):\n")
print(len(collection))
print("\nNUMBER OF DUPLICATES FOUND WITH LSH (SPARK):\n")
print(len(enum))
print("\nTIME OF THE EXECUTION (LSH - SPARK):\n")
print(end-start)
print("\nNUMBER OF NEAR-DUPLICATE PAIRS FOUND WITH COMPARISON OVER SHINGLES SETS:\n") 
print(len(enum_comparison))
print("\nTIME OF THE EXECUTION FOR THE PREVIOUS TASK:\n") 
print(end_2-start_2)
print("\nIntersection size between lsh and this method:\n") 
print(intersection)

print("\nWRITING THE RESULTS INTO FILE...\n")
with open(directory+"/duplicates_lsh_spark.tsv", "wt") as of: #write the obtained data into a .tsv file
	writer = csv.writer(of, delimiter="\t")
	writer.writerow(['id', 'link'])
	for d in enum:
		writer.writerow([d, links[d]])

print("\nDone. You can find the links of the duplicates under the file 'duplicates_lsh_spark.tsv'.\n")

#The results we found are the same obtained in exercise 2: check the .tsv files created at the end of each exercise. 
#So what was the difference? Spark was. Spark doesn't make us gain time,
#but it sure does, as a framework, make the job easier, since we were able to obtain the same results by writing less code - all the
#'for-loops' in exercise 2 and the structures exploited, were useless here since the Spark functions do all the job of collecting,
#grouping and iterating over the documents for us.
#Spark framework builds up as many processes as the number of objects in the RDD, so that each process can parse one and only one
#document.
#Since there is only one document at a time we have to care about as programmers, we write functions to parse one document at a time,
#so we don't really have to care about "the bigger picture" when we are implementing the LSH algorithm functions.
#The only problem I had was with implementing also the testing phase in Spark, which I think I did but asked the framework too much,
#since it kept running out of memory because of the very high number of combinations. I managed to make it work with some workaround,
#but still I thought that was too inefficient to be implemented officialy so I just reported it as a comment to make you see
#how I did it.