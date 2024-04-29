# @Author : Cheng Huang
# @Time   : 14:46 2024/4/20
# @File   : compgcn_embedding.py.py
# @Author : Cheng Huang
# @Time   : 15:11 2022/9/5
# @File   : transe_embedding.py
# This file is to get the transE embedding of the lmdb dataset and the dbpedia dataset
import os
import time

from pykeen.triples import TriplesFactory
from pykeen.losses import MarginRankingLoss
from pykeen.pipeline import pipeline
from pykeen.optimizers import Adam
from pykeen.evaluation import RankBasedEvaluator
import torch
from pykeen.utils import set_random_seed
# from pykeen.pipeline import set_random_seed

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 只使用 GPU 0
specific_seed=42
# random.seed(specific_seed)
# np.random.seed(specific_seed)
# torch.manual_seed(specific_seed)
# torch.cuda.manual_seed_all(specific_seed)
# set_random_seed(specific_seed)

def get_embedding_CompGCN(path, training, testing, validation):

    # grid search to find the best hyper-parameter
    dbmodel = None
    if "dbpedia" in path:
        dbmodel = pipeline(
            model='CompGCN',
            training=training,
            testing=testing,
            validation=validation,
            model_kwargs=dict(
                embedding_dim=100,
            ),
            training_kwargs=dict(
                num_epochs=10,
                batch_size=128,
            ),
            stopper='early',  # 使用提前停止
            stopper_kwargs=dict(
                frequency=5,       # 每5个epoch检查一次性能
                patience=20,       # 如果20个epoch内没有改善，则停止
                relative_delta=0.01  # 改善至少需要1%
            )
        )
        # dbmodel.save_to_directory('model_complete_dbpedia_CompGCN_default_100/dbpedia_CompGCN_model')

    lmmodel = None
    if "lmdb" in path:
        lmmodel = pipeline(
            model='CompGCN',
            training=training,
            testing=testing,
            validation=validation,
            model_kwargs=dict(
                interaction="distmult",
                embedding_dim=100,
            ),
            training_kwargs=dict(
                num_epochs=10,  # 设置较大的epoch，期望通过提前停止来中断
                batch_size=128,
            ),
            stopper='early',  # 使用提前停止
            stopper_kwargs=dict(
                frequency=5,       # 每5个epoch检查一次性能
                patience=20,       # 如果20个epoch内没有改善，则停止
                relative_delta=0.01  # 改善至少需要1%
            )
        )

        # lmmodel.save_to_directory('model_compgcn_default_complete_lmdb_50/lmdb_CompGCN_model')


def get_embedding_TransE(path, training, testing, validation, lr, dim, fn, margin):

    # grid search to find the best hyper-parameter
    dbmodel = None
    if "dbpedia" in path:
        dbmodel = pipeline(
            model='TransE',
            training=training,
            testing=testing,
            validation=validation,
            training_loop='sLCWA',
            negative_sampler='basic',
            loss=MarginRankingLoss,
            loss_kwargs = dict(margin=margin),
            model_kwargs = dict(
                scoring_fct_norm = fn,
                embedding_dim=dim),
            training_kwargs=dict(
                num_epochs=300,  # 设置较大的epoch，期望通过提前停止来中断
                batch_size=128,
            ),
            optimizer=Adam,
            optimizer_kwargs=dict(lr=lr),
            stopper='early',  # 使用提前停止
            stopper_kwargs=dict(
                frequency=5,       # 每5个epoch检查一次性能
                patience=20,       # 如果20个epoch内没有改善，则停止
                relative_delta=0.01  # 改善至少需要1%
            )
        )
        # dbmodel.save_to_directory('/home/cheng/guess2/embedding/transe_embedding/dbpedia_transe_model_dim_{}_lr_{}_fn_{}_margin_{}'.format(dim, lr, fn, margin))

    lmmodel = None
    if "lmdb" in path:
        lmmodel = pipeline(
           model='TransE',
           training=training,
            testing=testing,
            validation=validation,
            training_loop='sLCWA',
            negative_sampler='basic',
            loss=MarginRankingLoss,
            loss_kwargs = dict(margin=margin),
            model_kwargs = dict(
                scoring_fct_norm = fn,
                embedding_dim=dim),
            training_kwargs=dict(
                num_epochs=10,  # 设置较大的epoch，期望通过提前停止来中断
                batch_size=128,
            ),
            optimizer=Adam,
            optimizer_kwargs=dict(lr=lr),
            stopper='early',  # 使用提前停止
            stopper_kwargs=dict(
                frequency=5,       # 每5个epoch检查一次性能
                patience=20,       # 如果20个epoch内没有改善，则停止
                relative_delta=0.01  # 改善至少需要1%
            )
        )

        # lmmodel.save_to_directory('/home/cheng/guess2/embedding/transe_embedding/lmdb_transe_model_dim_{}_lr_{}_fn_{}_margin_{}'.format(dim, lr, fn, margin))




def get_embedding_distmult(path, training, testing, validation, lr, dim, margin):

    # grid search to find the best hyper-parameter
    dbmodel = None
    if "dbpedia" in path:
        dbmodel = pipeline(
            model='DistMult',
            training=training,
            testing=testing,
            validation=validation,
            loss=MarginRankingLoss,
            loss_kwargs = dict(margin=margin),
            model_kwargs = dict( 
                embedding_dim=dim),
            training_kwargs=dict(
                num_epochs=10,  # 设置较大的epoch，期望通过提前停止来中断
                batch_size=128,
            ),
            optimizer=Adam,
            optimizer_kwargs=dict(lr=lr),
            stopper='early',  # 使用提前停止
            stopper_kwargs=dict(
                frequency=5,       # 每5个epoch检查一次性能
                patience=20,       # 如果20个epoch内没有改善，则停止
                relative_delta=0.01  # 改善至少需要1%
            )
        )
        # dbmodel.save_to_directory('/home/cheng/guess2/embedding/distmult_embedding/dbpedia_distmult_model_dim_{}_lr_{}_margin_{}'.format(dim, lr, margin))

    lmmodel = None
    if "lmdb" in path:
        lmmodel = pipeline(
            model='DistMult',
            training=training,
            testing=testing,
            validation=validation,
            loss=MarginRankingLoss,
            loss_kwargs = dict(margin=margin),
            model_kwargs = dict(
                embedding_dim=dim),
            training_kwargs=dict(
                num_epochs=10,  # 设置较大的epoch，期望通过提前停止来中断
                batch_size=128,
            ),
            optimizer=Adam,
            optimizer_kwargs=dict(lr=lr),
            stopper='early',  # 使用提前停止
            stopper_kwargs=dict(
                frequency=5,       # 每5个epoch检查一次性能
                patience=20,       # 如果20个epoch内没有改善，则停止
                relative_delta=0.01  # 改善至少需要1%
            )
        )

        # lmmodel.save_to_directory('/home/cheng/guess2/embedding/distmult_embedding/lmdb_distmult_model_dim_{}_lr_{}_margin_{}'.format(dim, lr, margin))

# This method is to evaluate the model
# using MRR and hits@10
def evluate_model(path, training, testing, validation, lr, dim, fn, margin):
    evaluator = RankBasedEvaluator()
    model = None

    if "dbpedia" in path:
        # TransE
        # model = torch.load("/home/cheng/guess2/embedding/transe_embedding/dbpedia_transe_model_dim_{}_lr_{}_fn_{}_margin_{}/trained_model.pkl".format(dim, lr, fn, margin)); 
        # DistMult
        model = torch.load("/home/cheng/guess2/embedding/distmult_embedding/dbpedia_distmult_model_dim_{}_lr_{}_margin_{}/trained_model.pkl".format(dim, lr, margin));
        # model = torch.load(os.path.join(os.getcwd(),"model_complete_dbpedia_CompGCN_default_100/dbpedia_CompGCN_model/trained_model.pkl"))
    else:
        # TransE
        # model = torch.load("/home/cheng/guess2/embedding/transe_embedding/lmdb_transe_model_dim_{}_lr_{}_fn_{}_margin_{}/trained_model.pkl".format(dim, lr, fn, margin)); 
        # DistMult
        model = torch.load("/home/cheng/guess2/embedding/distmult_embedding/lmdb_distmult_model_dim_{}_lr_{}_margin_{}/trained_model.pkl".format(dim, lr, margin));
        # model = torch.load(os.path.join(os.getcwd(),"model_compgcn_default_complete_lmdb_50/lmdb_CompGCN_model/trained_model.pkl"))
    result = evaluator.evaluate(
        model=model,
        mapped_triples=testing.mapped_triples,
        batch_size=1024,
        additional_filter_triples=[
            training.mapped_triples,
            validation.mapped_triples
        ]
    )
    return result.get_metric("meanreciprocalrank"), result.get_metric("hits@10")

    


def choose(path):
    # tf = TriplesFactory.from_path(path, create_inverse_triples=True)
    tf = TriplesFactory.from_path(path,create_inverse_triples=True)
    # split the data into training set, testing set, validation set
    training, testing, validation = tf.split([.8, .1, .1])
    # if "dbpedia" in path:
    #     get_embedding_TransE(path, training, testing, validation, 0.01, 100, 2, 1)
    # else:
    #     get_embedding_TransE(path, training, testing, validation, 0.001, 50, 1, 2)

    dims = [50, 100]
    lrs = [0.001, 0.01]
    fns = [1, 2]
    margins = [1, 2, 10]


    # for lr in lrs:
    #     for dim in dims:
    #         for margin in margins:
    # lr = 0.001
    # dim = 100
    # margin = 1
    # result = evluate_model(path, training, testing, validation, lr, dim, margin) 
    # print("ls = {}, dim = {}, margin = {}".format(lr, dim, margin), result) 
                  # for lr in lrs:

    ## Evaluate TransE model
    # for lr in lrs:
    #     for dim in dims:
    #         for fn in fns:
    #             for margin in margins:
    #                 result = evluate_model(path, training, testing, validation, lr, dim, fn, margin)
    #                 print("ls = {}, dim = {}, fn = {}, margin = {}". format(lr, dim, fn, margin), result)

    ## Evaluate DistMult model
    lr = 0.01    
    for lr in lrs:
        for dim in dims:
            for margin in margins:
                result = evluate_model(path, training, testing, validation, lr, dim, 1, margin)
                print("ls = {}, dim = {}, fn = {}, margin = {}".format(lr, dim, 1, margin), result)


if __name__ == '__main__':
    root = os.path.abspath(os.path.dirname(os.getcwd()))
    db_path = "/home/cheng/entity_summarization/complete_data/dbpedia/complete_extract_dbpedia.tsv"
    lm_path = "/home/cheng/entity_summarization/complete_data/lmdb/complete_extract_lmdb.tsv"
    # choose(db_path)
    choose(lm_path)



