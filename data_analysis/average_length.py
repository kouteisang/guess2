import os
import numpy as np

root = os.path.abspath(os.path.dirname(os.getcwd()))
path = os.path.join(root,"data","elist.txt")

f = open(path, 'r')

cnt_dbpedia = 0
cnt_lmdb = 0

total_dbpedia = 0
total_lmdb = 0

for line in f:
    _,dataset,_,_,_,tripleNum = line.rstrip("\n").split("	")
    if dataset == "dbpedia":
        cnt_dbpedia += 1
        total_dbpedia += int(tripleNum)
    elif dataset == "lmdb":
        cnt_lmdb += 1
        total_lmdb += int(tripleNum)

print("average dbpedia = ",total_dbpedia/cnt_dbpedia)
print("average lmdb = ",total_lmdb/cnt_lmdb)