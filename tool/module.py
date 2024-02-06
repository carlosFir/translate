import torch
import numpy as np
import pandas as pd
import datetime
import time
import copy
from tqdm import tqdm

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    '''
    position is the length of encoding
    d_model is the dimension of the model
    return a tensor (1, position, d_model)
    '''
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)

def create_padding_mask(seq, pad):
    '''
    pad is the token_ids for pad
    return a tensor (batch_size, 1, 1, seq_len)
    '''
    mask = torch.eq(seq, torch.tensor(pad)).to(torch.float32)   
    return mask[:, np.newaxis, np.newaxis, :]

def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones((size, size)), diagonal=1)
    return mask 


def create_mask(inp, targ, pad):
    '''
    inp is a tensor (b, inp_seq_len)
    targ is a tensor (b, targ_seq_len)
    pad is the token_id for pad

    return three masks:
    enc_padding_mask: (b, 1, 1, inp_seq_len)
    combined_mask: (b, 1, targ_seq_len, targ_seq_len)
    dec_padding_mask: (b, 1, 1, inp_seq_len)
    '''
    look_ahead_mask = create_look_ahead_mask(targ.shape[-1]).to(inp.device)
    dec_targ_padding_mask = create_padding_mask(targ, pad)

    enc_padding_mask = create_padding_mask(inp, pad)
    # print(look_ahead_mask.device, dec_targ_padding_mask.device)
    combined_mask = torch.max(look_ahead_mask, dec_targ_padding_mask)
    dec_padding_mask = create_padding_mask(inp, pad)

    return enc_padding_mask, combined_mask, dec_padding_mask



def scaled_dot_product_attention(q, k, v, mask=None):
    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    d_k = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_attention_logits = matmul_qk / torch.sqrt(d_k)

    if mask is not None:
        scaled_attention_logits += (mask * 1e-9)
    
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v)

    return output, attention_weights

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads

        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)

        self.final_linear = torch.nn.Linear(d_model, d_model)
    
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)
    
    def forward(self, q, k, v, mask):
        batch_size = q.shape[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        attention, weights = scaled_dot_product_attention(q, k, v, mask)
        attention = attention.transpose(1, 2)
        attention = attention.reshape(batch_size, -1, self.d_model)

        output = self.final_linear(attention)
        return output, weights

def point_wise_feed_forward_network(d_model, dff):
    feed_forward_net = torch.nn.Sequential(
        torch.nn.Linear(d_model, dff),
        torch.nn.ReLU(),
        torch.nn.Linear(dff, d_model)
    )
    return feed_forward_net

class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = MultiHeadAttention(self.d_model, self.num_heads)
        self.ffn = point_wise_feed_forward_network(self.d_model, self.dff) # linear + relu + linear

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        # self.layernorm3 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)

    def forward(self, x, mask):
        attn_out, _ = self.mha(x, x, x, mask)
        attn_out = self.dropout1(attn_out)
        out1 = self.layernorm1(attn_out + x)

        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(ffn_out + out1)

        return out2

class DecoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm3 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(rate)
        self.dropout2 = torch.nn.Dropout(rate)
        self.dropout3 = torch.nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        ffn_out = self.ffn(out2)
        ffn_out = self.dropout3(ffn_out)
        out3 = self.layernorm3(ffn_out + out2)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 d_model, 
                 num_heads,
                 dff,
                 input_vocab_size,
                 maximun_position_encoding,
                 rate=0.1):
        super(Encoder, self).__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
    

        self.embedding = torch.nn.Embedding(input_vocab_size, d_model)
        self.positional_encoding = positional_encoding(maximun_position_encoding, d_model)
        self.enc_layers = torch.nn.ModuleList([EncoderLayer(d_model, 
                                                            num_heads, 
                                                            dff,
                                                            rate) for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, mask):
        input_seq_len = x.shape[-1]

        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        pos_encoding = self.positional_encoding[:, :input_seq_len, :].to(x.device)

        x += pos_encoding
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x 

class Decoder(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 target_vocab_size,
                 maximun_position_encoding,
                 rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers
        self.d_model = d_model

        self.embedding = torch.nn.Embedding(target_vocab_size, d_model)
        self.positional_encoding = positional_encoding(maximun_position_encoding, d_model)
        self.dec_layers = torch.nn.ModuleList([DecoderLayer(d_model,
                                                            num_heads,
                                                            dff,
                                                            rate) for _ in range(num_layers)])
        self.dropout = torch.nn.Dropout(rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        targ_seq_len = x.shape[-1]
        attention_weights = {}

        x = self.embedding(x)
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        pos_encoding = self.positional_encoding[:, :targ_seq_len, :].to(x.device)
        x += pos_encoding
        x = self.dropout(x)

        for i in range(self.num_layers):
            x, attn_block1, attn_block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            
            attention_weights[f'decoder_layer{i + 1}_block1'] = attn_block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = attn_block2

        return x, attention_weights
    

class Transformer(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 input_vocab_size,
                 target_vocab_size,
                 pe_input,
                 pe_target,
                 rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               input_vocab_size,
                               pe_input,
                               rate)
        
        self.decoder = Decoder(num_layers,
                               d_model,
                               num_heads,
                               dff,
                               target_vocab_size,
                               pe_target,
                               rate)
        
        self.lm_head = torch.nn.Linear(d_model, target_vocab_size)

    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, enc_padding_mask)

        dec_output, attention_weights = self.decoder(tar, enc_output, look_ahead_mask, dec_padding_mask)

        final_output = self.lm_head(dec_output)

        return final_output, attention_weights

def mask_loss_function(real, pred, pad):
    '''
    real is a tensor (b, seq_len)
    pred is a tensor (b, seq_len, vocab_size)
    pad is the token_id for pad
    '''
    loss_object = torch.nn.CrossEntropyLoss(reduction='none')
    _loss = loss_object(pred.transpose(-1, -2), real)
    
    mask = torch.logical_not(real.eq(pad)).type(_loss.dtype)
    
    _loss *= mask

    return _loss.sum() / mask.sum().item()

def mask_accuracy_function(real, pred, pad):
    '''
    real is a tensor (b, seq_len)
    pred is a tensor (b, seq_len, vocab_size)
    pad is the token_id for pad
    '''
    _pred = pred.argmax(dim=-1)
    _corrects = _pred.eq(real)

    mask = torch.logical_not(real.eq(pad))
    _corrects *= mask

    return _corrects.sum().float() / mask.sum().item()


def train_step(model, inp, targ, optimizer, pad):
    targ_inp = targ[:, :-1]
    targ_real = targ[:, 1:]

    # print(inp.device, targ.device)
    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ_inp, pad)

    model.train()
    optimizer.zero_grad()

    prediction, _ = model(inp, targ_inp, enc_padding_mask, combined_mask, dec_padding_mask)

    loss = mask_loss_function(targ_real, prediction, pad)
    accuracy = mask_accuracy_function(targ_real, prediction, pad)

    loss.backward()
    optimizer.step()

    return loss.item(), accuracy.item()

def validate_step(model, inp, targ, pad):
    targ_inp = targ[:, :-1]
    targ_real = targ[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_mask(inp, targ_inp, pad)

    model.eval()
    with torch.no_grad():
        prediction, _ = model(inp, targ_inp, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = mask_loss_function(targ_real, prediction, pad)
        accuracy = mask_accuracy_function(targ_real, prediction, pad)

    return loss.item(), accuracy.item()

def print_bar():
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('====================', now_time, '====================')

def train_model(model, 
                optimizer, 
                epochs, 
                train_dataloader, 
                val_dataloader, 
                print_every, 
                save_dir,
                save_name,
                pad,
                device,
                ngpu=1,):
    
    starttime = time.time()
    df_history = pd.DataFrame(columns=['epoch', 'loss', 'acc', 'val_loss', 'val_acc'])
    print_bar()
    

    best_acc = 0.
    for epoch in range(1, epochs + 1):

        loss_sum = 0.
        acc_sum = 0.
        length = len(train_dataloader)
        with tqdm(total=length) as t:

            for step, (inp, targ) in enumerate(train_dataloader):
                inp = inp.to(device)
                targ = targ.to(device)

                loss, acc = train_step(model, 
                                    inp, 
                                    targ, 
                                    optimizer,
                                    pad)
                loss_sum += loss
                acc_sum += acc

                '''t.set_description(desc="Epoch {}".format(epoch), refresh=True)
                t.set_postfix(steps=step, loss=loss_sum / (step + 1), acc=acc_sum / (step + 1))
                t.update(1)'''

                if step % print_every == 0:
                    print(f'Epoch {epoch} | Step {step} | Loss {loss_sum / (step + 1):.4f} | Accuracy {acc_sum / (step + 1):.4f}')

            val_loss_sum = 0.
            val_acc_sum = 0.
            
            for val_step, (inp, targ) in enumerate(val_dataloader):
                inp = inp.to(device)
                targ = targ.to(device)

                val_loss, val_acc = validate_step(model, inp, targ, pad)
                val_loss_sum += val_loss
                val_acc_sum += val_acc
            

            record = (epoch, loss_sum / step, acc_sum / step, val_loss_sum / val_step, val_acc_sum / val_step)
            df_history.loc[epoch - 1] = record

            print('EPOCH: {}, LOSS: {:.4f}, ACC: {:.4f}, VAL_LOSS: {:.4f}, VAL_ACC: {:.4f}'.format(
                record[0], record[1], record[2], record[3], record[4]
            ))

            print_bar()
    

    checkpoint = save_dir + save_name

    if inp.device.type == 'cuda' and ngpu > 1:
        model_sd = copy.deepcopy(model.module.state_dict())
    else:
        model_sd = copy.deepcopy(model.state_dict())

    torch.save({
        'loss': loss_sum / step,
        'epoch': epoch,
        'net': model_sd,
        'opt': optimizer.state_dict()
    }, checkpoint)

    print('Finishing training...')
    
    endtime = time.time()
    time_elapsed = endtime - starttime
    print('Training completed in {:.0f}m {:0f}s'.format(time_elapsed //60, time_elapsed % 60))

    return df_history

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)






if __name__ == '__main__':
    x = positional_encoding(50, 512)
    print(x.shape)

    y = create_padding_mask(torch.tensor([[1, 2, 3, 0, 0, 0, 0, 0]]), 1)
    print(y)

    z = create_look_ahead_mask(8)
    print(z)

    combined_mask = torch.max(z, y)
    print(combined_mask)



















