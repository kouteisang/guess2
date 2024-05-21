

from experiment.write_result import get_res
from experiment.KmeansExp import get_res_k_means


if __name__ == '__main__':
    '''
        :parameter1 : dataset DBPEDIA or LMDB
        :parameter2 : k(number of cluster)
        :parameter3 : m(fuzzy parameter)
    '''

    ks = [3, 4, 5, 6, 7, 8, 9]
    ms = [2, 5, 9]
    type = "distmult"

    ## Test Guess
    for k in ks:
        for m in ms:
            get_res("dbpedia", k, m, type)
            get_res("lmdb", k, m, type)
