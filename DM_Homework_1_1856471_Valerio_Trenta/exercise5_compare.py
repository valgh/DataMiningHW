#EXERCISE 5
#now, our strategy is to load and read the .tsv dataset, extract only the columns we are interested in - locations, prices - and compute our results
import pandas as pd
#we exploit pandas since it lets us select only those columns of the file we are interested in!
directory = "/home/valeriop/Scrivania/sapienza/DataMining/Homework1/" #to be changed, depending on where you run the code!
with open(directory+"dataset_kijiji.tsv") as f:
	df = pd.read_csv(f, sep='\t', names=['links', 'titles', 'timestamps', 'locations', 'descriptions', 'prices'])
	locations = df['locations'].to_dict()
	prices = df['prices'].to_dict()
#now we merge them into a single dictionary
import locale
locale.setlocale(locale.LC_NUMERIC,"it_IT.utf-8") #BE CAREFUL: run locale -a command on your terminal and change this according to your environment!
for key in prices: #we want them to be float values
	if prices[key]!="Contatta l'utente" and prices[key]!="prices":
		prices[key]=locale.atof(prices[key][:-2])
#now each euro price has been converted into a float number
print(locations)
print(prices)
output={}
#now for each apartment we count the locations and sum all the prices for each location: 2617 announcements in dataset
for key in range(0, 2617): #this range should be changed according to the number of announcements we have in the dataset, if dataset also changes!
	if locations[key] not in output:
		if prices[key]!="Contatta l'utente" and prices[key]!="prices":
			output[locations[key]]=(1, prices[key])
	elif prices[key]!="Contatta l'utente" and prices[key]!="prices":
		output[locations[key]]=(output[locations[key]][0]+1, output[locations[key]][1]+prices[key])

print(output)

for key in output:
	output[key] = (output[key][0], output[key][1]/output[key][0])

print("\nFINAL OUTCOME: {'LOCATION ': (NUMBER OF ANNOUNCEMENTS IN LOCATION, AVERAGE PRICE)}:\n")

print(output)
