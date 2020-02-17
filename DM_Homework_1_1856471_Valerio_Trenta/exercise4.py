#PROCESS DATA LAST.FM
#first, we need to process the .tsv file and turn data into an inverted index: track id -> list of users id
directory = "/home/valeriop/Scrivania/sapienza/DataMining/Homework1/lastfm-dataset/" #TO BE CHANGED, DEPENDS ON WHERE THE CODE RUNS!
import csv
inverted_index = {}
list_of_tracks = []
#we are going to build an inverted index in this loop:
with open(directory+"output_final_dataset.tsv") as fd:
	reader = csv.reader(fd, delimiter = "\t")
	for row in reader:
		user = row[0]
		tracks = row[1]
		list_of_tracks = tracks.split(" ")
		for track in list_of_tracks:
			if track in inverted_index:
				inverted_index[track].append(user)
			else:
				inverted_index[track] = [user]

print("\nThis is the obtained inverted index:\n")
print(inverted_index)

#now that we have our inverted index, we can finally process it with our algorithm to compute the results
print("Counting number of users per track...")
results = {}
for track in inverted_index:
	results[track] = len(inverted_index[track])
	print("+")
print("Done.")
#now we sort the results to obtain the most listened tracks at the top of the dictionary
import collections
print("\nSorting results...\n")
sorted_results = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
sorted_results_twenties = sorted_results[:20]
#better to handle this as a dictionary
sorted_results_twenties_dict = collections.OrderedDict(sorted_results_twenties)
print("\nFirst twenty most listened tracks:\n")
print(sorted_results_twenties)
#we take these first twenty most listened tracks...
sorted_results_twenties_dict = collections.OrderedDict(sorted_results_twenties)
final_results = {}
print("\nComputing tuples of most listened tracks...\n")
#...and compute the requested top 10 tuples
for track1 in sorted_results_twenties_dict:
	for track2 in sorted_results_twenties_dict:
		if track1!=track2:
			print("-")
			if (track2, track1) not in final_results:
				print("+")
				final_results[(track1, track2)] = len(inverted_index[track1])+len(inverted_index[track2])
			else:
				print("Nope.")
		else:
			print("Nope.")

#now for the final output
print("\nDone.\n")
final_results_sorted = sorted(final_results.items(), key=lambda kv: kv[1], reverse=True)
print("\nTop 10 Tuples of most-listened tracks among tracks that have been listened to at least 20 times:\n")
print(final_results_sorted[:10])
