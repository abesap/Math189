import similaritymeasures
import dtw
import os
import json 
import k_means
import pandas as pd
import numpy as np
import stockclass
import matplotlib.pyplot as plt 
import matplotlib as mpl
with open('industry.json') as json_file:  
		indus = json.load(json_file)
with open('sector_comp.json') as json_file:  
		times = json.load(json_file)
def predict(name, index, plot = False):
    """ INPUTS
            1) values - the price/volume data for the stock of interest
            2) name - the name of the stock, will be grabbed to get the real values" 
        OUTPUTS
            1) The array of predicted values up till the end of the quarter"""
    sector = indus[name]
    with open(sector+'/alike.json') as json_file:  
        alike = json.load(json_file)
    similar = alike[name]
    stock = stockclass.stock(name)
    if index>=len(stock.days):
        return [0],[0]
    simpleprediction = simple_pred(stock.days[index]['close'],sector)
    basal=np.zeros_like(simpleprediction)
    if plot: 
        plt.plot(simpleprediction,linestyle = "--", color= 'cornflowerblue', label = "Simple Prediction")
    for x in similar:
        temp = stockclass.stock(x[0])
        pred_temp = simple_pred(temp.days[index]['close'],sector)
        basal += 1/x[1] * pred_temp
    toreturn = np.linalg.norm(simpleprediction)*simpleprediction+ np.linalg.norm(basal)*(basal/basal[0])/(np.linalg.norm(simpleprediction)+np.linalg.norm(basal))
    toreturn = toreturn/toreturn[0]
    if plot:
        plt.plot(basal/basal[0],linestyle = ":",color= "blue", label = "DTW Prediction")
        plt.plot(toreturn,color= "darkblue", label = "Hybrid Prediction")
        plt.plot(np.array(stock.quarter[index]['close']),color = 'red', label = "Actual Behavior")
        plt.legend(loc = 0)
        plt.xlabel("Days")
        plt.ylabel("Normalized Price")
        plt.title("Predicted Behavior of "+ name)
        plt.savefig
        if not os.path.exists(sector+"/"+name+"/"):
            os.mkdir(sector+"/"+name+"/")
        
        plt.savefig(sector+"/"+name+"/"+str(index)+".png")
        plt.close()
    return toreturn, stock.quarter[index]['close']
def accuracy(name,lis= []):
    for index in range(6):
        pred,real = predict(name,index)
        if pred[-1] != 0:
            lis.append(pred[-1]-real[-1])
    return lis
def simple_pred(value,sector):
    q_clusters = np.load(sector+'/quartercluster.npy')
    markov_days = pd.read_pickle(sector+'/days_8'+'.pkl')
    label = day_label(value,sector)
    m,n = q_clusters.shape
    index = markov_days.loc[label].idxmax()
    a = q_clusters[index]
    return a 
def plot_sector(sector):
    q_clusters = np.load(sector+'/quartercluster.npy')
    plt.style.use('seaborn-white')
    for x in q_clusters:
        plt.plot(x)
    plt.xlabel("Days")
    plt.grid(True)
    plt.ylabel("Adjusted Cost")
    plt.title(sector+" Clusters")
    plt.savefig(sector+'/k_means_clusters.png')
    plt.close()
def day_label(value, sector):
    d_clusters=np.load(sector +'/daycluster.npy')
    l = [np.linalg.norm(value-d_clusters[y]) for y in range(len(d_clusters))]
    label= (np.argsort(l)).item(0)
    return label
stocks = []
for x in times:
    if x != "Financials":
        stocks.append(times[x])
lis = []
for x in stocks:
    for y in x:
        lis = accuracy(y,lis) 
toreturn= []
for x in lis: 
    if not (np.isnan(x)): 
        toreturn.append(x) 