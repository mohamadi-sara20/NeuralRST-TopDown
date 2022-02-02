from re import sub
import sys, time, os
import copy, math
import operator
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from in_out.instance import CResult, Gold
from models.metric import Metric
from modules.embedding import Embedding
from modules.layer import *
import pickle

sys.path.append(".")

import argparse
import torch

from in_out.reader import Reader
from in_out.util import load_embedding_dict, get_logger
from in_out.preprocess import create_alphabet
from in_out.preprocess import batch_data_variable
from models.vocab import Vocab
from models.metric import Metric
from models.config import Config
from models.architecture import MainArchitecture
from train_rst_parser import predict

def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config_path', required=True)
    args = args_parser.parse_args()
    config = Config(None)
    print('config', args.config_path)
    config.load_config(args.config_path)
    
    logger = get_logger("RSTParser (Top-Down) RUN", config.use_dynamic_oracle, config.model_path)
    word_alpha, tag_alpha, gold_action_alpha, action_label_alpha, relation_alpha, nuclear_alpha, nuclear_relation_alpha, etype_alpha = create_alphabet(None, config.alphabet_path, logger)
    vocab = Vocab(word_alpha, tag_alpha, etype_alpha, gold_action_alpha, action_label_alpha, relation_alpha, nuclear_alpha, nuclear_relation_alpha)
    
    network = MainArchitecture(vocab, config) 
    network.load_state_dict(torch.load(config.model_name))

    if config.use_gpu:
        network = network.cuda()
    network.eval()

    
    
    # logger.info('Reading dev instance, and predict...')
    # reader = Reader(config.dev_path, config.dev_syn_feat_path)
    # dev_instances  = reader.read_data()
    # predict(network, dev_instances, vocab, config, logger)

    # logger.info('Reading test instance, and predict...')
    # reader = Reader(config.test_path, config.test_syn_feat_path)
    # test_instances = reader.read_data()
    # predict(network, test_instances, vocab, config, logger)


    # Testing one single instance
    # sample 1
    # fname = 'gum/gum_sample.txt'
    # fbiaff = 'gum/gum_vecs'
    # reader = Reader('./data/rst3.test38', './data/SyntaxBiaffine/test.conll.dump.results')

    directory = 'pcc_unseen/'
    listdir = os.listdir(directory)
    print(listdir)
    for l in listdir:
        if os.path.isdir(directory + l):
            print(l)
            files = os.listdir(directory + l)
            fbiaff = f'{directory}/{l}'
            for f in files:
                print('=======processing' + f)
                if f.endswith('.prep'):
                    fname = f'{directory}/{l}/{f}'
                    reader = Reader(fname, fbiaff)
                    test_instances = reader.read_data()
                    indices = np.array((range(0, 1)))
                    subset_data = batch_data_variable(test_instances, indices, vocab, config)
                    # span = [[(0, 138)]]
                    predictions_of_subtrees_ori, prediction_of_subtrees = network.forward(subset_data)
                    subtrees = []
                    print('3333', prediction_of_subtrees)
                    print(len(subtrees))
                    for subtree in prediction_of_subtrees[0].subtrees:
                        print('5555', subtree)
                        subtrees.append([subtree.nuclear, subtree.edu_start, subtree.edu_end, subtree.relation])
                    with open(f'output/{f}.predict', 'w') as g:
                        print(subtrees, f'{f}.predict', file=g) 
                        # os.system(f'{subtrees} >> output/{f}.txt')
                        
                    print('4444', subtrees)

if __name__ == '__main__':
    main()
