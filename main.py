

from experiment.write_result import get_res


if __name__ == '__main__':
    '''
        :parameter1 : dataset DBPEDIA or LMDB
        :parameter2 : k(number of cluster)
        :parameter3 : m(fuzzy parameter)
    '''

    ks = [3, 4, 5, 6, 7, 8, 9]
    ms = [2, 5, 9]
    type = "compgcn"

    for k in ks:
        for m in ms:
            # get_res("dbpedia", k, m, type)
            get_res("lmdb", k, m, type)



# java -jar esummeval_v1.2.jar /Users/huangcheng/Documents/ESBasedonSimilarity/ESBM_benchmark_v1.2 /Users/huangcheng/Documents/ESBasedonSimilarity/res_data/esbm/k_3_m_2

# java -jar esummeval_v1.2.jar /Users/huangcheng/Documents/ESBasedonSimilarity/ESBM_benchmark_v1.2 /Users/huangcheng/Documents/ESBasedonSimilarity/res_data/esbm_plus/complete_k_3_m_2

# java -jar esummeval_v1.2.jar /Users/huangcheng/Documents/ESBasedonSimilarity/ESBM_benchmark_v1.2 /Users/huangcheng/Documents/ESBasedonSimilarity/res_data/KMeansClustering
# java -jar esummeval_v1.2.jar /Users/huangcheng/Documents/ESBasedonSimilarity/ESBM_benchmark_v1.2 /Users/huangcheng/Documents/ESBasedonSimilarity/res_data/esbm_plus_test/complete_k_3_m_2
