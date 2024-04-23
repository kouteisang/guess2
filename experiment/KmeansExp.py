# @Author : Cheng Huang
# @Time   : 13:56 2024/1/12
# @File   : k-means.py

import os
import torch
from pykeen.triples import TriplesFactory
from embedding.get_embedding import get_embedding_representation
from KMeansBased.Kmeans import kmeans_clustering_entity_summarization



def kmeans_store(top_k, type, id, name):
    root = os.path.abspath(os.path.dirname(os.getcwd()))+"/guess2/"
    folder_path = os.path.join(root, "res_data", "KMeansClustering", name)
    folder = os.path.exists(folder_path)
    if not folder:
        os.makedirs(folder_path)
    if not os.path.exists(os.path.join(folder_path, str(id))):
        os.makedirs(os.path.join(folder_path, str(id)))

    file_origin = os.path.join(root, "data", name+"_data", str(id), "{}_desc.nt".format(id))

    if type == "top":
        res_path = os.path.join(folder_path, str(id),"{}_top{}.nt".format(id, len(top_k)))
    if type == "rank":
        file_origin_list = []
        res_path = os.path.join(folder_path, str(id), "{}_rank.nt".format(id))
        with open(file_origin, 'r') as f:
            for line in f:
                line = line[:-1]
                file_origin_list.append(line)
        res = open(res_path, 'w')
        for ele in top_k:
            res.write(file_origin_list[ele]+'\n')
        res.close()
        f.close()
        return

    res = open(res_path, 'w')
    cnt = -1
    with open(file_origin, 'r') as f:
        for line in f:
            cnt = cnt + 1
            line = line[:-1]
            if cnt in top_k:
                res.write(line+'\n')
    res.close()
    f.close()


def get_res_k_means(name):
    root = os.path.abspath(os.path.dirname(os.getcwd())) + "/guess2/"
    if name == "dbpedia":
        all_file = os.path.join(root, "data_analysis", "dbpedia", "dbpedia_all.txt")
        # model_path = os.path.join(root,"embedding","model_dbpedia","dbpedia_transe_model","trained_model.pkl")
        model_path = os.path.join(root,"embedding","dbpedia","dbpedia_CompGCN_model","trained_model.pkl")
        file_base = os.path.join(root,"data_analysis", "dbpedia")
        file_path = []
        for i in range(1,101):
            file_path.append({os.path.join(file_base,"{}_desc.nt".format(i)):i})
        for i in range(141, 166):
            file_path.append({os.path.join(file_base,"{}_desc.nt".format(i)):i})

    elif name == "lmdb":
        print(root)
        all_file = os.path.join(root, "data_analysis", "lmdb", "lmdb_all.txt")
        model_path = os.path.join(root, "embedding", "lmdb", "lmdb_CompGCN_model", "trained_model.pkl")
        file_base = os.path.join(root,"data_analysis", "lmdb")
        file_path = []
        for i in range(101,141):
            file_path.append({os.path.join(file_base,"{}_desc.nt".format(i)):i})
        for i in range(166, 176):
            file_path.append({os.path.join(file_base,"{}_desc.nt".format(i)):i})

    model = torch.load(model_path)
    tf = TriplesFactory.from_path(all_file)

    for file in file_path:
        key = list(file)[0]  # file path
        value = file[key]  # id
        embedding_rep = get_embedding_representation(tf, model, key)
        top_5, top_10 = kmeans_clustering_entity_summarization(embedding_rep)
        kmeans_store(top_5, "top", value, name)
        kmeans_store(top_10, "top", value, name)