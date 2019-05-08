import dataloading
import json
import os
import matplotlib.pyplot as plt 
import copy
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D 
#import similaritymeasures
import numpy as np 
daterange = ((2014,12,1),(2018,2,1))
with open('industry.json') as json_file:  
    industry = json.load(json_file)
class stock:
    def __init__(self,name,dr = daterange):
        self.name = name #string
        self.sector = industry[name] #string also
        self.valss = dataloading.query(dr, ticker= [name]) #temp variable
        self.quarter = dataloading.quarter_trunate(self.valss)#List of dataframes (date, close, volume)
        for y in range(len(self.quarter)):
            self.quarter[y]['range']= [x for x in range(len(self.quarter[y]))] 
        self.predvals = {}
        self.days = copy.deepcopy(self.quarter)
        for y in range(len(self.days)):
            self.days[y]= self.days[y].iloc[0:5] 
        self.polybasis = []
        self.quarter_curve = []
        #self.pred_values=[self.qpoly[x])for x in range(len(self.quarter))]
    def find_pred_vals(self):
        model = make_pipeline(PolynomialFeatures(10), Ridge())
        for z in range(len(self.quarter)):
            x = self.quarter[z]['range'][:,np.newaxis]
            y =self.quarter[z]['close']
            basis = np.polyfit(self.quarter[z]['range'],y,10)
            model.fit(x,y)
            y_pred = model.predict(x)
            self.predvals[z]=(x,y,y_pred)
            self.polybasis.append(basis)
            self.quarter_curve.append(y_pred)
            self.quarter[z]['pred_val']= y_pred
        return self
    def graph(self):
        self.find_pred_vals()
        if not os.path.exists("plots/"+self.name+"/"):
            os.mkdir("plots/"+self.name+"/")
        for z in list(self.predvals.keys()):
            x,y,y_pred = self.predvals[z]
            plt.plot(x,y)
            plt.xlabel("Days")
            plt.ylabel("Adjusted Cost")
            plt.plot(x,y_pred)
            plt.title(str(self.name+self.sector))
            plt.savefig(str("plots/"+self.name+"/"
            +self.name+str(z)))
            plt.close()
    def frechet_dist(self, rhs):
        frech = np.empty((len(self.quarter_curve),len(rhs.quarter_curve)))
        x = 0
        y = 0 
        for ls in self.quarter_curve:
            lh_size = len(ls)
            y = 0 
            for rs in rhs.quarter_curve:
                rh_size = len(rs)
                cut = min(lh_size,rh_size)
                frech[x,y] = similaritymeasures.frechet_dist(ls,rs)
                y+= 1
            x+=1
        return frech
    def basis_dist(self, rhs):
        basis = np.empty((len(self.polybasis),len(rhs.polybasis)))
        x = 0
        y = 0 
        for ls in self.polybasis:
            y = 0 
            for rs in rhs.polybasis:
                basis[x,y] = np.linalg.norm(ls-rs)
                y+= 1
            x+=1
        return basis
    def yield_(self, dic):
        for z in range(len(self.polybasis)):
            x = self.quarter[z]['close'][-1]
            if x > 1.1:
                dic['+++'].append(np.poly1d(self.polybasis[z]))
            elif x> 1.05:
                dic['++'].append(np.poly1d(self.polybasis[z]))
            elif x>1:
                dic['+'].append(np.poly1d(self.polybasis[z]))
            elif x>.95:
                dic['-'].append(np.poly1d(self.polybasis[z]))
            elif x>.90:
                dic['--'].append(np.poly1d(self.polybasis[z]))
            else:
                dic['---'].append(np.poly1d(self.polybasis[z]))
    def markov_stats(self,dic):
        self.yield_(dic)
        
        
"""lst = ['+++','++', '+', '---','--', '-']
dic = {x:[] for x in lst}
for x in dataloading.tickere:
    a= stock(x)
    a.find_pred_vals()
    a.yield_(dic)"""



            


