import dataloading
import json
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
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
        quarter_features= PolynomialFeatures(degree=15)
        day_features= PolynomialFeatures(degree=3)
        for y in range(len(self.quarter)):
            self.quarter[y]['range']= [x for x in range(len(self.quarter[y]))] 
        #self.qpoly = [quarter_features.fit_transform(self.quarter[x].close)) for x in range(len(self.quarter))]
        #self.dpoly = [day_features.fit_transform(self.quarter[x].close)) for x in range(len(self.quarter))]
##### CURRENTLY NEED TO ADD A TIME AXIS FOR FITTING REGRESSION WITH RESPECT TO TIME
