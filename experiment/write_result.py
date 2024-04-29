import os

import torch
from pykeen.triples import TriplesFactory
from greedy.entropy_search import cluster_entropy
from embedding.get_embedding import get_embedding_representation
from soft_clustering.fuzzy_k_means import FCM



def store(top_k, type, id, k, m, name):
    root = os.path.abspath(os.path.dirname(os.getcwd()))+"/guess2/"
    folder_name = "k_" + str(k) + "_m_" + str(m)
    folder_path = os.path.join(root, "res_data", "transe_bad", folder_name, name)
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

def get_res(name, k, m, type):
    '''

    :param name: "dbpedia" or "lmdb"
    :param k: number of cluster, hyperparameter
    :param m: the fuzzy k-means m
    :param type: embedding method transe or distmult
    :return:
    '''
    root = os.path.abspath(os.path.dirname(os.getcwd())) + "/guess2/"
    if name == "dbpedia":
        all_file = os.path.join(root, "data_analysis", "dbpedia", "dbpedia_all.txt")

        if type == 'transe':
            model_path = "/home/cheng/guess2/embedding/transe_embedding/dbpedia_transe_model_dim_100_lr_0.01_fn_1_margin_1/trained_model.pkl"
        elif type == 'distmult':
            model_path = "/home/cheng/guess2/embedding/distmult_embedding/dbpedia_distmult_model_dim_100_lr_0.001_margin_1/trained_model.pkl"
        elif type == "compgcn":
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

        if type == 'transe':
            model_path = "/home/cheng/guess2/embedding/transe_embedding/lmdb_transe_model_dim_100_lr_0.01_fn_1_margin_1/trained_model.pkl"
        elif type == 'distmult':
            model_path = "/home/cheng/guess2/embedding/distmult_embedding/lmdb_distmult_model_dim_100_lr_0.001_margin_1/trained_model.pkl"
        elif type == "compgcn":
            model_path = os.path.join(root, "embedding", "lmdb", "lmdb_CompGCN_model", "trained_model.pkl")
        file_base = os.path.join(root,"data_analysis", "lmdb")
        file_path = []
        for i in range(101,141):
            file_path.append({os.path.join(file_base,"{}_desc.nt".format(i)):i})
        for i in range(166, 176):
            file_path.append({os.path.join(file_base,"{}_desc.nt".format(i)):i})

    print("**",model_path)
    model = torch.load(model_path)
    tf = TriplesFactory.from_path(all_file)
    for file in file_path:
        key = list(file)[0] # file path
        value = file[key] # id
        embedding_rep = get_embedding_representation(tf, model, key)
        t = FCM(embedding_rep, k, m, 0.001).forward()
        res = cluster_entropy(t) # entropy based method
        top_5 = res[:5]
        top_5.sort()
        store(top_5, "top", value, k, m, name)
        top_10 = res[:10]
        top_10.sort()
        store(top_10, "top", value, k, m, name)
        store(res, "rank", value, k, m, name)

