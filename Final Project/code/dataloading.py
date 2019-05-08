import quandl
import json
import numpy as np
from dateutil.parser import parse
import datetime as dt
import pandas as pd
import pandas_datareader as web
import sklearn
import sklearn.preprocessing
import sklearn.cluster
import sklearn.mixture 
import matplotlib.pyplot as plt 
from scipy.stats import norm
from tqdm import tqdm
import sys
quandl.ApiConfig.api_key = 'SwncpGcT1qFyFb3FHU2w'
data =quandl.get("FSE/VOS_X", returns= "numpy")
start= dt.datetime(2015,1,3)
end = dt.datetime(2018,12,31)
#### Steps create array of stock prices ####, 
tickere = ('NKE', 'MRK', 'WMT', 'JNJ', 'MMM', 'HD'
,'XOM','GS', 'DIS', 'DWDP','IBM', 'CVX', 'UNH', 
'VZ','CAT','UTX','V', 'TRV', 'CSCO', 'AXP', 'PG')

database = 'iex'
def query(DateRange,ticker= tickere, database=database):
    ''' 
        ticker is a  stock tickers that 
            that we are going to query
        DateRange is the list of a start and stop date
            that we are going to query
        database is the data base, from quandl that
            we want to direct it to
        returns a list of strings to query
    '''
    start= dt.datetime(DateRange[0][0],DateRange[0][1],DateRange[0][2])
    end = dt.datetime(DateRange[1][0], DateRange[1][1], DateRange[1][2])
    if type(ticker) == str: 
        ticker = [ticker]
    to_return = {}
    for x in ticker:
        to_return[x] = web.DataReader(x,database, start, end)
    return to_return
### Normalize values###
def normalize (query):
    dictionary = query
    toreturn = []
    for ticker in list(dictionary.keys()):
        templist = []
        count = 0
        for x in dictionary[ticker].axes[0]:
            count +=1
            if (count%28==1):
                val = dictionary[ticker].at[x, 'open' ]
            temp = dictionary[ticker].at[x, 'open' ]
            count-=1
            if not (np.isnan(temp)):
                templist.append(temp/val)
                count+=1
            if (count%28 == 0):
                toreturn.append(np.array(templist))
    prenorm = np.array(toreturn)
    #sklearn.preprocessing.normalize(prenorm,copy = False)
    return [x.tolist() for x in prenorm]
def search_for_quarters(que):
    tuple_quarters= [('03','31'),
        ('06','30'),('09','30'),('12','30')]
    toreturn = []
    for x in que[list(que.keys())[0]].axes[0]: 
        year,month,day = (x.strip('-').split('-'))
        if (month,day) in tuple_quarters:
            toreturn.append(x)
    return toreturn
def quarter_trunate(que):
    vals = que
    toreturn = search_for_quarters(vals)
    stock_list = []
    for ticker in list(vals.keys()):
        if ticker in vals:
            temp = vals[ticker]
            vals = temp.iloc[:,3:5]
            for y in range(len(toreturn)-1):
                normalized = vals.loc[toreturn[y]:toreturn[y+1]]
                a = normalized.apply(lambda x: x/x.iloc[0], axis = 0)
                stock_list.append(a)
        else:
            pass
    return stock_list
daterange = ((2014,12,1),(2018,2,1))
def local_quarter(query):
    """Inputs a query object and returns a np array of arrays representing 
        the close price and volume, relative to the initial value, for 10
        days around the quarterly times """
    contquart = search_for_quarters(query)
    change = dt.timedelta(days = 5)
    toreturn = []
    for x in contquart:
        year,month,day = (x.strip('-').split('-'))
        quarter=dt.date(int(year),int(month),int(day))
        advance = quarter+change
        before = quarter-change
        ayear = advance.year
        amonth = advance.month
        aday = advance.day
        byear = before.year
        bmonth = before.month
        bday = before.day
        before= str(byear)+'-'+"{:02d}".format(bmonth)+'-'+"{:02d}".format(bday)
        after= str(ayear)+'-'+"{:02d}".format(amonth)+'-'+"{:02d}".format(aday)
        toreturn.append((before,after))
    stock_list = []
    for ticker in list(query.keys()):
        if ticker in query:
            temp = query[ticker]
            query = temp.iloc[:,3:5]
            for y in range(len(toreturn)):
                normalized = query.loc[toreturn[y][0]:toreturn[y][1]]
                a = normalized.apply(lambda x: x/x.iloc[0], axis = 0)
                stock_list.append(a)
        else:
            pass
    return stock_list

### Run K means on this array ###
def kMeans(prenorm, clusters = 8):
    prenorm = [x for x in prenorm if len(x) != 28]
    kmeans = sklearn.cluster.KMeans(n_clusters = clusters).fit(prenorm)
    return kmeans
def GMM(prenorm,clusters = 8):
    gmm = sklearn.mixture.GaussianMixture(n_components=clusters, covariance_type='full').fit(prenorm)
    return gmm
### Plot K means outputs ###
def plotlike(prenorm, kmeans):
    for y in range(len(prenorm)):
        plt.style.use('ggplot')
        plt.plot(prenorm[y])
    plt.xlabel("Time(days)")
    plt.ylabel("Price(Normalized)")
    plt.title("Dow Jones Industrial")
    plt.savefig("sp500.png") 
    plt.show()
    
    for x in range(len(kmeans.cluster_centers_)):
        for y in range(len(prenorm)):
            if kmeans.labels_[y] == x:
                plt.plot(prenorm[y],'r')
        plt.plot(kmeans.cluster_centers_[x],"bo")
        plt.show()
    for x in kmeans.cluster_centers_:
        plt.plot(x)
    plt.xlabel("Time(days)")
    plt.ylabel("Price(Normalized)")
    plt.title("Cluster Centers")
    plt.savefig("Clutcentsp.png")
    plt.show()
### Compare to human observer patterns ###
def predict(days,kmeans):
    ###For Each partially predict by Sum of Squares
    #Create a dictionary for each of the clusters
    #and index through it, also need to make sure to store the max value of the array.
    temp = kmeans.cluster_centers_
    length = len(days)
    truncated = [x[:length]for x in temp]
    subtracted = [np.subtract(x,days) for x in truncated]
    errors = [np.linalg.norm(x) for x in subtracted]
    #print(errors)
    return np.argmin(errors)

        
    ##while days != []:
def load(filename):
    """ Inputs a filename and returns a list of their symbols"""
    tickers =[]
    with open(filename)as f:
        for x in f:
            x =x[:-1]
            tickers.append(x)
    return tickers  
def industries(filename, outfilename):
    """ Inputs a filename and returns a list of their symbols"""
    tickers ={}
    with open(filename)as f:
        for x in f:
            x = x.split(",")
            tickers[x[0]]= x[1][:-1]
    with open(outfilename+".json", 'w') as outfile:
        json.dump(tickers, outfile)
def truncate(prenorm, length=5):
    toreturn = [x[:length]for x in tqdm(prenorm)]
    return toreturn
    #takes in prenorm array and chops it down for testing
def predictedcenters (truncated, kmeans):
    clusters = []
    for x in tqdm(truncated):
        clusters.append(predict(x,kmeans))
    return clusters
def calcaccuracy (pred_labels, real_labels, numclusters):
    """
		HINT:
			1) Get the predict lables using predict defined above
			2) Accuracy = probability of correct prediction
			3) Precision = probability of true label being 1 given that the
			   predicted label is 1
			4) Recall = probablity of predicted label being 1 given that the
			   true label is 1
			5) F-1 = 2*p*r / (p + r), where p = precision and r = recall

		NOTE: Please use the variable given for final returned results.
        """
    totcount = len(pred_labels)
    scores = [pred_labels[x]==real_labels[x] for x in range(totcount)]
    return sum(scores)/totcount
def accur(fname, noclust, daterange):
    tick= load(fname) 
    queried = query(daterange, tick)
    prenorm = normalize(queried)
    kmeans = kMeans(prenorm, clusters= noclust)
    real_labels= kmeans.labels_
    values = []
    for x in tqdm(range(len(kmeans.cluster_centers_[1]))):
        days = truncate(prenorm, length=x)
        pred_labels = predictedcenters(days, kmeans)
        values.append(calcaccuracy(pred_labels, real_labels, noclust))
    return values,kmeans,prenorm
def compare(fname, noclust, daterange):
    tick= load(fname) 
    queried = query(daterange, tick)
    prenorm = normalize(queried)
    gmm = GMM(prenorm,noclust)
    kmeans = kMeans(prenorm, clusters= noclust)
    real_labels= kmeans.labels_
    values = []
    for x in tqdm(range(len(kmeans.cluster_centers_[1]))):
        days = truncate(prenorm, length=x)
        pred_labels = predictedcenters(days, kmeans)
        values.append(calcaccuracy(pred_labels, real_labels, noclust))
    return values,gmm,kmeans,prenorm
def plotaccuracy (values):
    plt.plot(values)
    plt.title("Predication Accuracy Relative to Days Data Included")
    plt.xlabel("Time(Days)")
    plt.ylabel("Accuracy")
    plt.savefig("Accuracy.png")
def noisedistributions(kmeans, prenorm):
    noise = []
    for x in range(len(prenorm)):
        center = kmeans.cluster_centers_[kmeans.labels_[x]]
        nois = prenorm[x]-center
        for y in nois:
            if y != 0:
                noise.append(y)
    plt.hist(noise,bins= 256, density=True)
    mu, std = norm.fit(noise)
    a = np.std(noise)
    print(a)
    plt.xlabel("Error")
    plt.ylabel("Density")
    plt.title("Distribution of Error")
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    plt.savefig("hist.png", dpi = 300)
    plt.show()
    errbound = 1.96*a
    count = 0
    sig = []
    for x in kmeans.cluster_centers_:
        if abs(x[0]-x[-1])>errbound:
            sig.append(count)
            plt.plot(x,'r', linewidth = 2)
        else:
            plt.plot(x, 'black', linewidth = 2)
        count +=1 
    print (sig)
    val = len(list(filter(lambda x: x in sig, kmeans.labels_)))
    count = val/len(kmeans.labels_)
    plt.title("Significant Clusters")
    plt.savefig("SigClus.png", dpi = 300)
    plt.show()
    return noise,a,count
 
def plotgmm(gmm):
    plt.style.use('ggplot')
    for x in gmm.means_:
        plt.plot(x)
    plt.title("Mean Clusters")
    plt.xlabel("Time(days)")
    plt.ylabel("Value(normalized)")
    plt.savefig("gmmcent.png")
    plt.show()
def gmmnoisedistributions(gmm, prenorm):
    noise = [[] for x in range(len(gmm.means_))]
    percentage = [0 for x in range(len(gmm.means_))]
    labels = gmm.predict(prenorm)
    for x in range(len(prenorm)):
        center = gmm.means_[labels[x]]
        nois = prenorm[x]-center
        for y in nois:
            if y != 0:
                noise[labels[x]].append(y)
                percentage[labels[x]]+=1
    
    print(noise)
    val = np.sum(percentage)
    perc = [x/val for x in percentage]
    return noise,perc  
def plotnoise(noise,gmm):
    plt.style.use('ggplot')
    labels = []
    print(range(len(noise)))
    for x in range(len(noise)):
        print (x)
        label = "gmmcenter"+ str(x)+ ".png"
        plt.title("Cluster #"+str(x))
        plt.xlabel("Error")
        plt.ylabel("Density")
        mu, std = norm.fit(noise[x])
        errbound = 1.96*std
        if abs(gmm.means_[x][-1]-1)-errbound >0:
            print ("a")
            plt.hist(noise[x],color = 'red', bins = 30, density=True)
            labels.append(1)
        else:
            print (labels)
            plt.hist(noise[x],color = 'blue', bins = 30, density=True)
            labels.append(0)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        plt.savefig(label)
        plt.show()
    return labels
def plotsignif(gmm,labels):
    plt.style.use('ggplot')
    for x in range(len(labels)):
        if labels[x] == 1:
            plt.plot(gmm.means_[x], 'red', label = "DD")
        else:
            plt.plot(gmm.means_[x],'black') 
            print(x)  
    plt.title("Significant Clusters Using Gaussian Mixture")   
    plt.xlabel("Time(days)")
    plt.ylabel("Value(normalized)")
    plt.savefig("sigfig.png")
    plt.show()
def accuracy(kmean, difnorm):
    labels = kmean.predict(difnorm)
    tovalidate = []
    allpos=np.sum([1 for x in difnorm if x[-1]> 1])
    allneg=np.sum([1 for x in difnorm if x[-1]< 1])
    trueneg = 0
    falseneg = 0
    truepos = 0
    falsepos = 0
    for x in kmean.cluster_centers_:
        if x[-1] > 1:
            tovalidate.append(1)
        else:
            tovalidate.append(0)
    for x in range(len(labels)): 
        finalvalue = difnorm[x][-1]
        label = labels[x]
        predictedval = tovalidate[label]
        if predictedval:
            predpos +=1
        else: predneg +=1
    if finalvalue <1:
        if predictedval == 0: 
            trueneg+=1
        if predictedval != 0:
            falseneg+=1
    if finalvalue >1:
        if predictedval == 0:
            falsepos +=1
        if predictedval != 0:
            truepos +=1
    return allpos,allneg,predpos,predneg,trueneg,falseneg
### Label patterns, then run supervised learning ### 
### Predict trends in certain futures ###

