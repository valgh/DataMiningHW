#exercise2.py
import numpy as np
import math
from random import gauss
import time
import matplotlib.pyplot as plt
#setting variables
#didn't use all these variables reported down there in the comments. I exploited only some combinations and then other combinations,
#they can be found in the report .pdf.
k = 50 #100, 200
n = 1000 #10000, 100000
d = k #100*k, 100*k^2
stdev = math.sqrt(1/k) #1/sqrt(k),0.5
num_points = k*n
num_dimensions = d+k
data = np.ndarray(shape=(num_points, num_dimensions), dtype=np.float32, order='F') #create the n-dimensional array with the size we need


def build_dataset():
	print("\nBuilding the dataset...\n")
	start_dataset = time.time()
	for i in range(k*n):
		for j in range(d):
			data[i][k+j] = gauss(0,stdev) #here we are storing the Gaussians
		for j in range(k): #here we are storing the identity matrix which also "clusters" each Gaussian
			if i%k==j:
				data[i][j] = 1 #put point in cluster j if i mod k == j.
			else:
				data[i][j] = 0 #point is not in cluster j
	end_dataset = time.time()
	print("\n++++++++++++++++++++++DATASET BUILT:++++++++++++++++++++++\n")
	print(data)
	print("\n++++++++++++++++++++++BUILT IN:++++++++++++++++++++++\n")
	print(end_dataset-start_dataset)

#from here, we are going to recover the centroids of each cluster for the original ground clustering.
#We'll need this later, when trying to understand how much of the original ground clustering
#was retrieved by k means and k-means + pca.	

#recover the ground clustering
def ground_clustering(data): 
	print("\n++++++++++++++++++++++RECOVERING ORIGINAL GROUND CLUSTERING:++++++++++++++++++++++\n")
	ground_cluster = {}
	for i in range(k*n):
		for j in range(k):
			if i%k == j:
				if j in ground_cluster:
					ground_cluster[j].append(data[i])
				else:
					ground_cluster[j] = [data[i]]
	print("\nDONE.\n")
	return ground_cluster

#PCA algorithm, as it was provided, through SVD
#set m as close as possible to k, say m>=k.
#works fine, but not for big data!
def PCA_():
	m = k #set m exactly equal to k
	u,s,vh = np.linalg.svd(data, full_matrices=True)
	smat = np.zeros((k*n, d+k), dtype=np.float16)
	smat[:s.size, :s.size] = np.diag(s)
	for i in range(0, s.size): #here we are setting the smallest singular values to zero
		if i>m-1:
			s[i] = 0
	smat[:d+k, :d+k] = np.diag(s)
	projected = np.dot(u, np.dot(smat, vh))
	return projected

#for high values of k and n, the matrices that are created in the above function are just too "big" to be handled.
#We don't need those matrices as provided in that function, we can "cut" the parts that are not needed, this is what
#the next function does.

def PCA_truncated_SVD():
	m = k #set m exactly equal to k
	u,s,vh = np.linalg.svd(data, full_matrices=False)
	smat = np.zeros((m*n, d+m), dtype=np.float16)
	smat[:m, :m] = np.diag(s[:m]) #instead of looping through the s array, just exploit the part we need of it
	projected = np.dot(u[:, :m], np.dot(smat[:m, :m], vh[:m, :])) #the dot products can be now computed with truncated matrices/arrays
	return projected

#the two previous PCA functions did not actually implement dimensionality reduction. This one does.

def PCA_dim():
  #we need to normalize one of our components!
  for i in range(k*n):
    for j in range(k):
      if i%k == j:
        data[i][j] = data[i][j]/stdev
  m = k #set m exactly equal to k
  u,s,vh = np.linalg.svd(data, full_matrices=False) 
  smat = np.zeros((m*n, d+m), dtype=np.float16)
  smat[:m, :m] = np.diag(s[:m]) #instead of looping through the s array, just exploit the part we need of it
  smat = smat[:m, :m] #here we truncate
  u = u[:, :m]
  transformed = np.dot(u, smat) #this is where we finally obtain a dimensionality reduction. Hopefully now, results will be better with PCA and we will not suffer from the curse of dimensionality
  return transformed



#K-MEANS ++ algorithm class
class KMeansPlusPlus(object):

	def __init__(self, k, input_data):
		self.k = k
		self.input = input_data
		self.centroids = []

	#squared euclidean distance between two points
	def distance(self, p1, p2):
		return np.sum(np.power(np.subtract(p1,p2), 2))

	#get the clusters' current centers:
	def get_centers(self):
		return self.centroids

	def plus_plus_initialization(self):
		#first select a random centroid
		self.centroids.append(self.input[np.random.randint(self.input.shape[0]), :])
		#now for the rest of the centroids:
		for c in range(self.k-1):
			#distances of data points from nearest centroid:
			dist = []
			for p in range(self.input.shape[0]):
				point = self.input[p,:]
			#for point in self.input:
				d = math.inf
				#compute all the distances of point from previously selected centroid
				#and select the minimum one
				for j in range(len(self.centroids)):
					tmp = self.distance(point, self.centroids[j])
					d = min(d, tmp)
				dist.append(d)

			#next centroid as far as we can go
			dist = np.array(dist)
			next_ = self.input[np.argmax(dist), :]
			self.centroids.append(next_)
			dist = []
		return self.centroids

	def check_if_equal(self, curr, new):
		#checks termination condition
		if not curr:
			return False
		else:
			return np.array_equal(np.array(curr), np.array(new))

	def get_mean(self, in_cluster, dimensionality):
		cd = []
		pts = np.array(in_cluster)
		l = pts.shape[0]
		for dim in range(dimensionality):
			cd.append(np.sum(pts[:,dim])/l)
		mean = np.array(cd)
		return mean

	def k_means(self):
		current_config = []
		clusters = {}
		while self.check_if_equal(current_config, self.centroids) == False:
			#memorize the current configuration of the clusters'centers:
			current_config.clear()
			for c in self.get_centers():
				current_config.append(c)
			clusters.clear()
			for c in current_config:
				clusters[tuple(c)] = []
			#for every point in the dataset
			for point in self.input:
				d = math.inf
				current_centroid = tuple(current_config[0])
				for c in current_config:
					tmp_d = self.distance(point, c) #distance point-centroid
					if tmp_d<d:
						current_centroid = tuple(c)
						d = tmp_d
				clusters[current_centroid].append(point)
			#finished clustering, now recomputing the centroids for each cluster
			self.centroids.clear()
			dim = self.input[0].shape[0]
			for c in clusters:
				new_c = self.get_mean(clusters[c], dim)
				self.centroids.append(new_c)
		return clusters

#get max similarity between ground clusters and clusters found
def accuracy_clusters(kme, ground):
	output = {}
	for ground_cluster in ground:
		tmp_sim = 0
		for cluster in kme:
			similarity = len(set(list(map(tuple, kme[cluster]))).intersection(set(list(map(tuple, ground[ground_cluster])))))/len(set(list(map(tuple, kme[cluster]))).union(set(list(map(tuple, ground[ground_cluster])))))
			if max(tmp_sim, similarity) == similarity:
				tmp_sim = similarity
		output[ground_cluster] = tmp_sim
	return output

#"MAIN"

build_dataset() #"data" will be then constructed as we wish
clustering = KMeansPlusPlus(k, data)
print("\n+++++++++++++++INITIALIZATION PHASE FOR K-MEANS++ WITHOUT PCA:+++++++++++++\n")
start_init = time.time()
print("\nINIT PHASE OF K-MEANS++:\n")
clustering.plus_plus_initialization() #k-means++ init
print("\nDONE.\n")
end_init = time.time()
print("\nTIME TO INIT:\n")
print(end_init-start_init)
print("\n+++++++++++++++CLUSTERING PHASE FOR K-MEANS++ WITHOUT PCA:+++++++++++++\n")
print("\nOBTAINING CLUSTERS...\n")
start_clustering = time.time()
clusters = clustering.k_means() #k-means clustering after ++ init, if you wish to see the clusters, print this
#print(clustering.centroids) #which centroids did we get? Can print this to visualize them
end_clustering = time.time()
print("\nTIME TO CLUSTER:\n")
print(end_clustering-start_clustering)
print("\n+++++++++++++++END OF K-MEANS++ WITHOUT PCA:+++++++++++++\n")
print("\n+++++++++++++++INITIALIZATION PHASE FOR K-MEANS++ WITH PCA:+++++++++++++\n")
print("\nPERFORMING PCA...\n")
start_pca = time.time()
#data_pca = PCA_truncated_SVD()
data_pca = PCA_dim()
pca_clustering = KMeansPlusPlus(k, data_pca) #now the PCA_() function has applied PCA onto data
end_pca = time.time()
print("\nTIME TO PERFORM PCA:\n")
print(end_pca-start_pca)
print("\nINIT PHASE OF K-MEANS++:\n")
start_init_pca_kpp = time.time()
pca_clustering.plus_plus_initialization() #from now on, it's the same as before, but only on principal components
end_init_pca_kpp = time.time()
print("\nTIME TO INIT:\n")
print(end_init_pca_kpp-start_init_pca_kpp)
print("\nDONE.\n")
print("\n+++++++++++++++CLUSTERING PHASE FOR K-MEANS++ WITH PCA:+++++++++++++\n")
print("\nOBTAINING CLUSTERS...\n")
start_clustering2 = time.time()
clusters_pca = pca_clustering.k_means() #if you wish to see the clusters, print this
end_clustering2 = time.time()
print("\nTIME TO CLUSTER:\n")
print(end_clustering2-start_clustering2)
#the following is commented out since it was only needed to see how the "original pca" algorithm still gave bad results in terms
#of efficiency since it did not perform any dimensionality reduction
#print("\n+++++++++++++++INITIALIZATION PHASE FOR K-MEANS++ WITH PCA (SVD):+++++++++++++\n")
#print("\nPERFORMING PCA...\n")
#start_svd = time.time()
#data_svd = PCA_truncated_SVD()
#data_pca = PCA_dim()
# svd_clustering = KMeansPlusPlus(k, data_svd) #now the PCA_() function has applied PCA onto data
# end_svd = time.time()
# print("\nTIME TO PERFORM PCA (SVD):\n")
# print(end_svd-start_svd)
# print("\nINIT PHASE OF K-MEANS++:\n")
# start_init_svd_kpp = time.time()
# svd_clustering.plus_plus_initialization() #from now on, it's the same as before, but only on principal components
# end_init_svd_kpp = time.time()
# print("\nTIME TO INIT:\n")
# print(end_init_svd_kpp-start_init_svd_kpp)
# print("\nDONE.\n")
# print("\n+++++++++++++++CLUSTERING PHASE FOR K-MEANS++ WITH PCA:+++++++++++++\n")
# print("\nOBTAINING CLUSTERS...\n")
# start_clustering3 = time.time()
# clusters_svd = svd_clustering.k_means() #if you wish to see the clusters, print this
# end_clustering3 = time.time()
# print("\nTIME TO CLUSTER:\n")
# print(end_clustering3-start_clustering3)
print("\n+++++++++++++++COMPARING RESULTS WITH GROUND CLUSTERING:+++++++++++++++\n")
print("\nRECOVERING GROUND CLUSTER AND COMPARING...\n")
gc = ground_clustering(data)
acc_kme = accuracy_clusters(clusters, gc) #Jaccard similarity max for each cluster
print("\nDone.\n")
print("\nRECOVERING GROUND CLUSTER WITH PCA AND COMPARING...\n")
gc_pca = ground_clustering(data_pca)
acc_pca = accuracy_clusters(clusters_pca, gc_pca) #these results should actually be equal if not improved (at least when dimensionality is very high) wrt to the ones obtained before
print("\nDONE.")
#gc_svd = ground_clustering(data_svd)
#acc_svd = accuracy_clusters(clusters_svd, gc_svd)
print("\nRESULTS WITHOUT PCA:\n")
for c in acc_kme:
	print((c, acc_kme[c]), sep="\n")
print("\nRESULTS WITH PCA:\n")
for c in acc_pca:
	print((c, acc_pca[c]), sep="\n")
# print("\nRESULTS WITH SVD:\n")
# for c in acc_svd:
# 	print((c, acc_svd[c]), sep="\n")