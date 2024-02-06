from .module import *
import torch
from .loader import *


def greedy_decoder(model, encoder_input, decoder_input, tokenizer, MAX_LENGTH, pad):
    '''
    inp_sentence_ids: list of int
    tokenizer: instance of class Tokenizer()
    '''
    model.eval()


    # print(encoder_input.device, decoder_input.device)
    with torch.no_grad():

        for i in range(MAX_LENGTH + 1):
            enc_padding_mask, combined_mask, dec_padding_mask = create_mask(encoder_input, decoder_input, pad)
            # print(enc_padding_mask.device)

            predictions, _ = model(encoder_input, 
                                   decoder_input, 
                                   enc_padding_mask, 
                                   combined_mask, 
                                   dec_padding_mask)
            print(predictions)
            next_word = predictions[:, -1, :]
            next_ids = torch.argmax(next_word, dim=-1)
            # print(next_ids.shape, decoder_input.shape)
            
            if next_ids.squeeze().item() == tokenizer.word2index['<end>']:
                return decoder_input.squeeze(0), _
            
            decoder_input = torch.cat([decoder_input, next_ids.unsqueeze(0)], dim=1)
            
    return decoder_input.squeeze(0), _

    # print(encoder_input.shape, decoder_input)

if __name__ == '__main__':
    
    sentence = 'CCCCSS'

    tk = Tokenizer('vocab.txt')
 

   



