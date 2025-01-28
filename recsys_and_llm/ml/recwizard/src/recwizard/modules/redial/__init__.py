import logging

from . import params
from .autorec import AutoRec, ReconstructionLoss
from .beam_search import Beam, BeamSearch, get_best_beam
from .configuration_redial_gen import RedialGenConfig
from .configuration_redial_rec import RedialRecConfig
from .modeling_redial_gen import RedialGen
from .modeling_redial_rec import RedialRec
from .tokenizer_redial_gen import RedialGenTokenizer
from .tokenizer_redial_rec import RedialRecTokenizer
from .tokenizer_rnn import RnnTokenizer


def get_task_embedding(vocab, pretrained_emb="embeddings/glove.840B.300d.h5"):
    import h5py
    import numpy as np

    pretrained_embeddings = h5py.File(pretrained_emb)
    embedding_matrix = pretrained_embeddings["embedding"][()]
    pretrain_vocab = pretrained_embeddings["words_flatten"][()].decode().split("\n")
    pretrain_word2id = {word: ind for ind, word in enumerate(pretrain_vocab)}
    task_embeddings = []
    oov = 0
    for word in vocab:
        if word in pretrain_word2id:
            task_embeddings.append(embedding_matrix[pretrain_word2id[word]])
        else:
            oov += 1
            task_embeddings.append(np.zeros(300, dtype=np.float32))
    logging.info(f"{len(vocab)} words, {oov} oov")
    return np.stack(task_embeddings).astype(np.float32)
