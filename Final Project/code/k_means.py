""" Helper functions"""
import time
import numpy as np
import similaritymeasures
import copy
import json 
import sys
from tqdm import tqdm
import pandas as pd 
import stockclass
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
def k_means(X, k, eps=1e-100, max_iter=50, print_freq=5, prt = False):
	m,n = X.shape
	cost_list = []
	t_start = time.time()
	vals = np.random.choice(m,size=k)
	clusters = X[vals, :]
	label = np.zeros((m, 1)).astype(int)
	iter_num = 0
	while iter_num < max_iter:
		prev_clusters = copy.deepcopy(clusters)
		if prt:
			print("Minimizing the Frechet Distance", iter_num)
		for x in range(m):
			l = [similaritymeasures.frechet_dist(X[x,:],clusters[y]) for y in range(len(clusters))]
			#print (l)
			label[x] = (np.argsort(l)).item(0)
		if prt:
			print("Resetting the Centers")
		for x in range(k):
			locs = np.where(label==x)[0]
			if len(locs) > 0: 
				clusters[x,:] = X[locs].mean(axis = 0)
		cost = k_means_cost(X,clusters,label)
		cost_list.append(cost)

		if (iter_num + 1) % print_freq == 0:
			print('-- Iteration {} - cost {:4.4E}'.format(iter_num + 1, cost))
		if similaritymeasures.frechet_dist(prev_clusters, clusters) <= eps:
			print('-- Algorithm converges at iteration {} \
				with cost {:4.4E}'.format(iter_num + 1, cost))
			break
		iter_num += 1

	t_end = time.time()
	print('-- Time elapsed: {t:2.2f} \
		seconds'.format(t=t_end - t_start))
	return clusters, label, cost_list
def k_means_cost(X, clusters, label):
	"""	This function takes in the following arguments:
			1) X, the data matrix with dimension m x n
			2) clusters, the matrix with dimension k x 1
			3) label, the label of the cluster for each data point with
				dimension m x 1

		This function calculates and returns the cost for the given data
		and clusters.
		1) The total cost is defined by the sum of the frechet distance between each function and the function of 
        the cluster to which it is assigned 
	"""
	m, n = X.shape
	k = clusters.shape[0]
	X_cluster = clusters[label.flatten()]
	cost = similaritymeasures.frechet_dist(X, X_cluster).sum()
	return cost
def generate_and_pred(sector):
	"""INPUTS: 
			1) a string that is a investment sector 
	   OUTPUTS:
	   		1) A list of stocks, with predicted values comprising of 
			   elements of the S&P 500 that are in sector. 
				"""
	with open('sector_comp.json') as json_file:  
		sectors = json.load(json_file)
	list_o_stocks = [stockclass.stock(x) for x in sectors[sector]]
	list_o_stocks = [x.find_pred_vals() for x in list_o_stocks]
	return list_o_stocks
def markov(sector,k, plot = False, dump = False):
	""" INPUTS: 
			1) sector -a string that is a investment sector 
			2) k - the number of clusters with which to run K_means
		OUTPUTS:
			1) day_matrix - a markov model probability matrix predicting 
							quarter cluster based on the price of the 5 day
							surrounding the start of the quarter [k_quarters by k_day]
			2) vol_matrix - a markov model probability matrix predicting 
							quarter cluster based on the volume of the 5 day
							surrounding the start of the quarter [k_quarters by k_day]"""
	list_o_stocks = generate_and_pred(sector)
	days = []
	vols = []
	quarts = []
	for x in tqdm(list_o_stocks):
		for y in x.days:
			vols.append(y['volume'])
			days.append(y['close'])
		for y in x.quarter_curve:
			quarts.append(y)
	dl= min([len(x) for x in days])
	vl = min([len(x)for x in vols])
	ql = min([len(x) for x in quarts])
	print(ql)
	for x in range(len(days)):
		days[x] = days[x][:dl]
	for x in range(len(vols)):
		vols[x] = vols[x][:dl]
	for x in range(len(quarts)):
		quarts[x] = quarts[x][:ql]
	d_cent, d_lab, d_cost = k_means(np.array(days),k)
	v_cent, v_lab, v_cost = k_means(np.array(vols),k)
	q_cent, q_lab, q_cost = k_means(np.array(quarts),k, max_iter= 5, prt = True)
	day_matrix = pd.DataFrame(index = range(k) , columns =range(k), data=0)
	vol_matrix = pd.DataFrame(index = range(k) , columns =range(k), data=0)
	for x in range(len(q_lab)):
		day_matrix.loc[q_lab[x][0],d_lab[x][0]] +=1
	for x in range(len(q_lab)):
		vol_matrix.loc[q_lab[x][0],v_lab[x][0]] +=1
	for x in day_matrix:
		day_matrix.loc[x,:] = day_matrix.loc[x,:]**2/np.sum(day_matrix.loc[x,:]**2)
	for x in vol_matrix.index:
		vol_matrix[x] = vol_matrix[x]**2/np.sum(vol_matrix[x]**2)
	if plot:
		for x in q_cent:
			plt.plot(x)
		plt.xlabel("Days")
		plt.grid(True)
		plt.ylabel("Adjusted Cost")
		plt.title(sector+" Clusters")
		plt.savefig(sector+'/k_means_clusters.png')
		plt.close()
	if dump:
		vol_matrix.to_pickle(sector+'/volumes_'+str(k)+'.pkl')
		day_matrix.to_pickle(sector+'/days_'+str(k)+'.pkl')
		np.save((sector+"/quartercluster"),q_cent)
		np.save((sector+"/daycluster"),d_cent)
		np.save((sector+"/volcluster"),v_cent) 
	return day_matrix,vol_matrix, d_cent, v_cent, q_cent
"""sectorlist = ['Information Technology','Industrials','Health Care','Energy','Consumer Staples','Consumer Discretionary',
				'Communication Services', 'Materials','Real Estate', 'Utilities']
for sector in sectorlist:
	start = time.time()
	markov(sector,8, plot = True, dump = True)
	end = time.time()
	print(end-start) """


