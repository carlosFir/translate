import torchtext
from sklearn.model_selection import train_test_split
import tqdm
import torch
import numpy as np
from matplotlib import pyplot as plt
import os
import configparser
import warnings
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')

from tool.loader import *
from tool.preprocess import *
from tool.module import *
from tool.generate import *

# setting
# ---------------config parsing---------------
config = configparser.ConfigParser()
config.read('config.conf')

# ---------------cuda and device----------------


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = 'cuda:0'


# ---------------file to read --------------
data_path = '/home/Zhouyu/MODEL/translate/ustpo250k_list.txt'
vocab_path = '/home/Zhouyu/MODEL/translate/vocab.txt'

# ---------------hyper parameters------------

# data
MAX_LENGTH = int(config['data']['max_length']) # truncated to fixed length
BATCH_SIZE = int(config['data']['batch_size'])
# model
NUM_LAYERS = int(config['model']['num_layers'])
D_MODEL = int(config['model']['d_model'])
NUM_HEADS = int(config['model']['num_heads'])
DFF = int(config['model']['dff'])
DROPOUT = float(config['model']['rate'])
PE_INPUT = int(config['model']['pe_input'])
PE_TARGET = int(config['model']['pe_target'])
# train
EPOCHS = int(config['train']['epochs'])
SAVE_DIR = str(config['train']['save_dir'])
SAVE_NAME = str(config['train']['save_name'])
PRINT_TRAINSTEP_EVERY = int(config['train']['print_trainstep_every']) 
# optimizer
LR = float(config['optimizer']['lr'])
BETAS = (float(config['optimizer']['beta1']), float(config['optimizer']['beta2']))
EPS = float(config['optimizer']['eps'])  
# generate
GENERATE = str(config['generate']['on']) == 'True'
CKPT = str(config['generate']['ckpt'])
# -------------------------------preprocess-------------------------------
data = []
with open(data_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        data.append(line)

pairs = make_pairs(data)

pairs = [[pair[0], pair[1]] for pair in pairs if filterPair(pairs, MAX_LENGTH)]
# dataset and dataloader
train_pairs, test_pairs = train_test_split(pairs, test_size=0.3, random_state=1234)

train_dataset = ReaDataset(train_pairs, 'vocab.txt', MAX_LENGTH)
val_dataset = ReaDataset(test_pairs, 'vocab.txt', MAX_LENGTH)

train_dataloader = DataLoader(train_dataset, BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, 1)

input_vocab_size = 68
target_vocab_size = 68
pad = 1

# ---------------------------------model--------------------------------


model = Transformer(num_layers=NUM_LAYERS,
                    d_model=D_MODEL,
                    num_heads=NUM_HEADS,
                    dff=DFF,
                    input_vocab_size=input_vocab_size,
                    target_vocab_size=target_vocab_size,
                    pe_input=PE_INPUT,
                    pe_target=PE_TARGET,
                    rate=DROPOUT).to(device)


print('Model parameters: ', count_parameters(model))


if not GENERATE: 
    loss_object = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, eps=EPS)


    df_history = train_model(model, 
                            optimizer,
                            EPOCHS, 
                            train_dataloader, 
                            val_dataloader, 
                            PRINT_TRAINSTEP_EVERY, 
                            SAVE_DIR, 
                            SAVE_NAME,
                            pad,
                            device)
    
else:
    ckpt = torch.load(CKPT)
    model_sd = ckpt['net']

    model.load_state_dict(model_sd)
    tk = Tokenizer('vocab.txt')


    for (inp, targ) in val_dataloader:
        # print(inp, targ)
        encoder_input = inp[:, 1:].to(device)
        targ_output = targ[:, :-1].to(device)
        decoder_input = torch.tensor([tk.word2index['<start>']]).unsqueeze(0).to(device)
        # print(encoder_input.shape, decoder_input.shape)
        decoder_output, _ = greedy_decoder(model, encoder_input, targ_output, tk, 1, pad)
        print(decoder_output)
        
        break


    
    









