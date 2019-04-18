import dataloading
import json
import os
import matplotlib.pyplot as plt 
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from mpl_toolkits.mplot3d import Axes3D 
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
        self.days = dataloading.local_quarter(self.valss)
        quarter_features= PolynomialFeatures(degree=6)
        day_features= PolynomialFeatures(degree=3)
        for y in range(len(self.quarter)):
            self.quarter[y]['range']= [x for x in range(len(self.quarter[y]))] 
        for y in range(len(self.days)):
            self.days[y]['range']= [x for x in range(len(self.days[y]))]
        self.qpoly = [quarter_features.fit_transform(self.quarter[x]['range'][:,np.newaxis],self.quarter[x]['close']) for x in range(len(self.quarter))]
        self.dpoly = [day_features.fit_transform(self.quarter[x]) for x in range(len(self.quarter))]
        self.predvals = {}
        #self.pred_values=[self.qpoly[x])for x in range(len(self.quarter))]
    def find_pred_vals(self):
        model = make_pipeline(PolynomialFeatures(10), Ridge())
        for z in range(len(self.quarter)):
            x = self.quarter[z]['range'][:,np.newaxis]
            y =self.quarter[z]['close']
            model.fit(x,y)
            y_pred = model.predict(x)
            self.predvals[z]=(x,y,y_pred)
    def graph(self):
        self.find_pred_vals()
        if not os.path.exists("plots/"+self.name+"/"):
            os.mkdir("plots/"+self.name+"/")
        for z in list(self.predvals.keys()):
            x,y,y_pred = self.predvals[z]
            plt.plot(x,y)
            plt.plot(x,y_pred)
            plt.title(str(self.name+self.sector))
            plt.savefig(str("plots/"+self.name+"/"
            +self.name+str(z)))
            plt.close()


