#exercise1_clear_dataset.py
import pandas as pd
import os
#we exploit pandas since it lets us select only those columns of the file we are interested in!
directory = os.getcwd()
with open(directory+"/dataset_kijiji.tsv") as f:
	df = pd.read_csv(f, sep='\t', names=['links', 'titles', 'timestamps', 'locations', 'descriptions', 'prices'])
	links = df['links'].to_dict()
	titles = df['titles'].to_dict()
	timestamps = df['timestamps'].to_dict()
	locations = df['locations'].to_dict()
	descriptions = df['descriptions'].to_dict()
	prices = df['prices'].to_dict()

#eliminating the duplicates links already found in the dataset - they were due to the fact that some top announcements were
#repeated in the pages of the Kijiji web site:
already_seen = {} #auxiliary structure to keep track of the already seen links
for l in links:
	if links[l] not in already_seen:
		already_seen[links[l]] = [l]
	else:
		already_seen[links[l]].append(l)
for l in already_seen:
	if len(already_seen[l]) > 1: #if the link is repeated more than once in the dataset
		for key in already_seen[l]:
			if key != already_seen[l][0]: #keep the first link, and...
				del links[key] #...delete all the other entries for that link
				del titles[key]
				del timestamps[key]
				del locations[key]
				del descriptions[key]
				del prices[key]

#just to test this. We end up with 2380 announcements, so now we can print them to the new file
print(len(links))
print(len(titles))
print(len(timestamps))
print(len(locations))
print(len(descriptions))
print(len(prices))

import csv
with open(directory+"/dataset_kijiji_cleared.tsv", 'wt') as of:
	writer = csv.writer(of, delimiter="\t")
	writer.writerow(['links', 'titles', 'timestamps', 'locations', 'descriptions', 'prices'])
	for i in links:
		writer.writerow([links[i], titles[i], timestamps[i], locations[i], descriptions[i], prices[i]])

print("\nDone.\n")