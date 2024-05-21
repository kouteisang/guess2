import os

import torch
from pykeen.triples import TriplesFactory



def get_embedding_representation(tf, model, file_path):
    '''
    Args:
        tf: triple factory
        model: model
        file_path(str): file need to get the embedding

    Returns:
        emb_rep(list): return embedding representation
    '''
    emb_rep = []
    entity_embedding = model.entity_representations[0](indices=None).detach().to("cpu").numpy()
    relation_embedding = model.relation_representations[0](indices=None).detach().to("cpu").numpy()
    with open(file_path, 'r') as f:
        for line in f:
            h, r, t = line.split("\t")
            t = t.rstrip("\n")
            # get the corresponding id
            h, t = tf.entities_to_ids([h, t])
            r = tf.relations_to_ids([r])[0]
            # get the embedding
            h = entity_embedding[h].tolist()
            t = entity_embedding[t].tolist()
            r = relation_embedding[r].tolist()
            emb_rep.append(h+r+t)

    return emb_rep

