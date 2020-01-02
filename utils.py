
import os
import numpy as np
import pandas as pd
import pudb
import csv


def get_txt_paths(root):
    paths = []
    for file in os.listdir(root):
        if file.endswith(".txt"):
            paths += [os.path.join(root, file)]
    return paths
        

def get_fields(filename, fields):
    ret = {field : [] for field in fields}
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for idx, row in enumerate(csv_reader):
            if idx > 0:
                #date = row[0] #volume = row[5]
                for field in fields:
                    if field == 'closing_prices':
                        ret[field] += [float(row[4])]
                    elif field == 'volume': 
                        ret[field] += [float(row[5])]
    n_values =  len(ret[fields[0]])
    ret['n_values'] = n_values
    
    return ret
