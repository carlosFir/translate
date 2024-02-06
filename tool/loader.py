import torch

from torch.utils.data import DataLoader, Dataset 

class Tokenizer():
    '''
    vocab is a dictionary with keys as the words and values as the index
    '''
    def __init__(self, vocab_txt):

        self.vocab = self.build_vocab(vocab_txt)
        self.word2index = self.vocab
        self.index2word = list(self.vocab.keys())
        self.vocab_size = len(self.vocab)

        self.split_char = lambda x: [char for char in x]


    def build_vocab(self, vocab_txt):
        result_dict = {}
        with open(vocab_txt, 'r') as file:
            for line_number, line in enumerate(file):
                line = line.strip()
                result_dict[line] = line_number
        return result_dict
    
    def encode(self, sentence, max_length):

        sentence = self.split_char(sentence)
        sentence = ['<start>'] + sentence + ['<end>']
        sentence_ids = [self.word2index[word] for word in sentence]

        pad_ids = self.word2index['<pad>']
        sentence_ids.extend([pad_ids] * max_length)

        return sentence_ids[:max_length]
    
    def decode(self, sentence_ids):
        sts = ''
        for ids in sentence_ids:
            if ids >= self.vocab_size:
                sts += '<unk>'
                continue

            word = self.index2word[ids]
            sts += word
        
        return sts
    
class ReaDataset(Dataset):
    '''
    data is SMLIES list like [[src, targ], [...], ...]
    vocab_file is the path to the vocab.txt
    max_length is the max length of the sentence
    '''
    def __init__(self, data, vocab_file, max_length):
        super(ReaDataset, self).__init__()
        
        self.tokenizer = Tokenizer(vocab_file)
        self.data = self.pad_and_token(data, max_length)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def pad_and_token(self, data, max_length):
        result = []
        for pair in data:
            #print(pair)
            src = torch.tensor(self.tokenizer.encode(pair[0], max_length))
            # print(src)
            targ = torch.tensor(self.tokenizer.encode(pair[1], max_length))
            # print(targ)
            result.append([src, targ])
        return result



if __name__ == '__main__':
    data = [['Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>Cl.NC1(CCl)CCCC1.O=[N+]([O-])c1ccc(N=C=S)c2c1CCCC2>',
             'O=[N+]([O-])c1ccc(N=C2NC3(CCCC3)CS2)c2c1CCCC2'],
             ['c2c1CCCC2>',
             '(CCCC3)CS2)c2c1CCCC2']]
    
    dataset = ReaDataset(data, 'vocab.txt', 100)
   
    dataloader = DataLoader(dataset, 1)
    for batch in dataloader:
        src, targ = batch[0], batch[1]
        print(src, targ.shape)
        break

        
