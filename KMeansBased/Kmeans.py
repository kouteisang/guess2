
from sklearn.cluster import KMeans
from scipy.spatial import distance


def calculate_res(center, embedding_rep):
    length = len(embedding_rep)
    t_dis = 99999999
    res = 0
    for i in range(length):
        dst = distance.euclidean(center, embedding_rep[i])
        if dst < t_dis:
            t_dis = dst
            res = i

    return res

def kmeans_clustering_entity_summarization(embedding_rep):


    # calculate the top 5
    kmeans5 = KMeans(n_clusters=5).fit(embedding_rep)
    centers_top5 = kmeans5.cluster_centers_
    res_top5 = []
    for i in range(5):
        res = calculate_res(centers_top5[i], embedding_rep)
        res_top5.append(res)


    # calculate the top 10
    kmeans10 = KMeans(n_clusters=10).fit(embedding_rep)
    centers_top10 = kmeans10.cluster_centers_
    res_top10 = []
    for i in range(10):
        res = calculate_res(centers_top10[i], embedding_rep)
        res_top10.append(res)

    return res_top5, res_top10
