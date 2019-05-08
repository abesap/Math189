""" Implement a model that measures the scaling with respect to
Dynamic Time warping"""
import stockclass
import similaritymeasures
import numpy as np 
import pandas as pd 
from tqdm import tqdm
import k_means
import matplotlib.pyplot as plt
import json
def compare(a,b):
    vals = []
    if len(a.quarter) is not len(b.quarter):
        return 500
    for i in range(len(a.quarter)):
        
        if len(b.quarter[i])>= 3 and len(b.quarter[i])>= 4:
            temp_a = a.quarter[i].iloc[:,[2,3]]
            temp_b = b.quarter[i].iloc[:,[2,3]]
            dtw, alignment = similaritymeasures.dtw(temp_a,temp_b, metric= 'minkowski')
            vals.append(dtw)
        else: return vals
    return vals
def dtw_comparision(pols):
    pols = [x.find_pred_vals() for x in pols]
    names = [x.name for x in pols]
    returne= pd.DataFrame(index = names , columns =names)
    for y in tqdm(range (len(pols))):
        x = y
        while x<len(pols):
            temp = np.linalg.norm(compare(pols[x], pols[y]))
            returne.loc[names[x],names[y]]= temp
            returne.loc[names[y],names[x]]= temp
            x+= 1
    return returne,names
def plot_comparisons(sector):
    pols = k_means.generate_and_pred(sector)
   
    vals,names = dtw_comparision(pols)
    m,n = vals.shape
    for x in range(m):
        for y in range(n):
            a = vals.iloc[x,y]
            if a < 500  and a!= 0 :
                l.append(a)
    plt.hist(l,bins = m)
    plt.axvline(x = 9)
    plt.savefig(sector+"DTW.png")
    plt.close()
def find_sig_comparisons(sector, plot = False):
    pols = k_means.generate_and_pred(sector)
    vals,names = dtw_comparision(pols)
    dic = {x:[] for x in names}
    for x in names:
        for y in names:
            if x != y and vals.loc[x,y] < 9:
                dic[x].append((y,1/vals.loc[x,y]))
    if plot:
        l = []
        m,n = vals.shape
        for x in range(m):
            for y in range(n):
                a = vals.iloc[x,y]
                if a < 500  and a!= 0 :
                    l.append(1/a)
        plt.hist(l,bins = m)
        plt.axvline(x = 9)
        plt.title("Dynamic Time Warpings"+ sector)
        plt.savefig(sector+"/"+"DTW.png")
        plt.close()
    return dic
def dump_sig_comp(sector, fname):
    dic=find_sig_comparisons(sector,plot=True)
    with open(sector+"/"+fname+".json", 'w') as outfile:
        json.dump(dic, outfile)
sectorlist = ['Information Technology','Industrials','Health Care','Energy','Consumer Staples','Consumer Discretionary',
               'Communication Services', 'Materials','Real Estate', 'Utilities']




