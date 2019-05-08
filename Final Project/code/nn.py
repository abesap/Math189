import json 
import numpy as np
import pandas as pd 
import stockclass as s
from tqdm import tqdm
with open('sector_comp.json') as json_file:  
    sectors = json.load(json_file)
def generate_frechet_matrix(sector):
    returne= {}
    list_o_stocks = [s.stock(x) for x in sectors[sector]]
    list_o_stocks = [x.find_pred_vals() for x in list_o_stocks]
    for x in list_o_stocks:
        to_return = []
        for rhs in list_o_stocks[1:]:
            to_return.append(x.frechet_dist(rhs))
        returne[x.name]= to_return
        list_o_stocks = list_o_stocks[1:]
    return returne
def generate_basis_matrix(sector):
    returne= pd.DataFrame(index = sectors[sector], columns = sectors[sector], dtype= )
    list_o_stocks = [s.stock(x) for x in sectors[sector]]
    list_o_stocks = [x.find_pred_vals() for x in list_o_stocks]
    for x in tqdm(list_o_stocks):
        to_return = []
        for rhs in list_o_stocks[1:]:
           returne[x,rhs]= (x.basis_dist(rhs))
        list_o_stocks = list_o_stocks[1:]  
    return returne
sector = "Communication Services"
a = generate_basis_matrix(sector)

