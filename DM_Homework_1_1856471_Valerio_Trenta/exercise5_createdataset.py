#EXERCISE 5
import urllib3
from bs4 import BeautifulSoup
import time
#we retrieve a number of pages from target website with a for-loop
kijiji_url = "https://www.kijiji.it/case/affitto/roma-annunci-roma/?entryPoint=sb"
kijiji_url2 = "https://www.kijiji.it/case/affitto/roma-annunci-roma/?p="
web_pages={}
for x in range(1, 120): #notice that this range should be changed from time to time as the list of announcements changes over time!
	if x==1:
		http = urllib3.PoolManager()
		response = http.request('GET', kijiji_url)
		parsed_response = BeautifulSoup(response.data, 'lxml')
		web_pages[x]=parsed_response
	else:
		print("Parsing the pages...")
		new_url = kijiji_url2+str(x)+"&entryPoint=sb"
		http = urllib3.PoolManager()
		response = http.request('GET', new_url)
		parsed_response = BeautifulSoup(response.data, 'lxml')
		web_pages[x]=parsed_response
	time.sleep(2)

#now we are ready to parse them
only_links = {}
only_titles = {}
only_timestamps = {}
only_locations = {}
only_descriptions = {}
only_prices = {}
l=0
t=0
ts=0
lo=0
d=0
p=0
#only_dictionaries will be the final results
for key,wp in web_pages.items():
	start_tag = wp.find(id="search-result") #we take into account only the elements of the search-result, leaving out ads
	links = start_tag.find_all("a", class_="cta") #find links of announcements
	for link in links:
		only_links[l]=link.get('href')
		l+=1
	for title in links:
		only_titles[t]=title.contents[0].rstrip().lstrip() #get the titles
		t+=1
	timestamps = start_tag.find_all("p", class_="timestamp") #find timestamps
	for timestamp in timestamps:
		only_timestamps[ts]=timestamp.contents[0]
		ts+=1
	locations = start_tag.find_all("p", class_="locale") #find locations
	for location in locations:
		only_locations[lo]=location.contents[0]
		lo+=1
	descriptions = start_tag.find_all("p", class_="description") #find descriptions
	for description in descriptions:
		only_descriptions[d]=description.contents[0]
		d+=1
	prices = start_tag.find_all("h4", class_="price") #find prices
	for price in prices:
		only_prices[p]=price.contents[0]
		p+=1

print("\nEVERY ELEMENT WE ARE INTERESTED IN:\n")
print("\nLinks:\n")
print(only_links)
print("\nTitles:\n")
print(only_titles)
print("\nTimestamps:\n")
print(only_timestamps)
print("\nLocations:\n")
print(only_locations)
print("\nDescriptions:\n")
print(only_descriptions)
print("\nPrices:\n")
print(only_prices)

#now, to write everything in a csv, since we have 22 announcements per page and we have 119 pages, we have exactly 22*119 = 2618 announcements, but in the last page there are only 21, so:
import csv
directory = "/home/valeriop/Scrivania/sapienza/DataMining/Homework1/"
with open(directory+"dataset_kijiji.tsv", 'wt') as of:
	writer = csv.writer(of, delimiter="\t")
	writer.writerow(['links', 'titles', 'timestamps', 'locations', 'descriptions', 'prices'])
	for i in range(0,2616): #again, if the number of announcements that were retrieved changes, also this range should be changed!
		writer.writerow([only_links[i], only_titles[i], only_timestamps[i], only_locations[i], only_descriptions[i], only_prices[i]])

print("\nDone.\n")
#now we can load the .tsv file again and start computing our results
#this is done in the exercise5_compare.py file!
