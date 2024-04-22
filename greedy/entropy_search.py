from scipy.stats import entropy
import numpy as np

def cluster_entropy(cluster):
    '''
    :param cluster: the fuzzy cluster result
    :return res: the top K index
    '''
    l = cluster.shape[0]
    res = []
    res_dict = {}
    for i in range(l):
        print(cluster[i])
        res_dict[i] = entropy(cluster[i])

    sort_dict = sorted(res_dict.items(), key = lambda item:item[1], reverse=True)
    for item in sort_dict:
        res.append(item[0])

    print(res)

    return res

def cal_tf_idf(file):
    f = open(file, 'r')
    ff = open(file, 'r')
    s_list = []
    p_list = []
    o_list = []
    res_dict = {}

    total_item = 0
    cnt = 0

    for line in f:
        s, p, o = line.rstrip("\n").split("	")
        s_list.append(s)
        p_list.append(p)
        o_list.append(o)
        total_item += 1

    for line in ff:
        s, p, o = line.rstrip("\n").split("	")
        num_p = p_list.count(p)
        num_o = o_list.count(o)
        score = np.log(total_item / num_p) * (num_o/total_item)
        # score = np.log(total_item / num_o) * (num_p/total_item)
        res_dict[cnt] = score
        cnt += 1

    return res_dict



def cluster_entropy_tf_idf(cluster, file, alpha):
    l = cluster.shape[0]
    res = []
    entropy_dict = {}

    tfidf_dict = cal_tf_idf(file)

    for i in range(l):
        entropy_dict[i] = entropy(cluster[i])

    sum_entropy = 0.0
    sum_tfidf = 0.0

    for i in range(l):
        sum_entropy += entropy_dict[i]
        sum_tfidf += tfidf_dict[i]

    for i in range(l):
        entropy_dict[i] = entropy_dict[i]/sum_entropy
        tfidf_dict[i] = tfidf_dict[i]/sum_tfidf
    #
    # print("sort_entropy_dict = ", sorted(entropy_dict.items(), key=lambda item: item[1], reverse=True))
    # print("sort_tfidf_dict = ", sorted(tfidf_dict.items(), key=lambda item: item[1], reverse=True))

    res_dict = {}

    for i in range(l):
        res_dict[i] = alpha * entropy_dict[i] + (1-alpha) * tfidf_dict[i]

    sort_dict = sorted(res_dict.items(), key=lambda item: item[1], reverse=True)
    for item in sort_dict:
        res.append(item[0])

    return res
