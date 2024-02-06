
    
def make_pairs(data):
    '''
    Data is a list of strings.
    Each string is like 'a>b>c' or 'a.s>b>c' where a and s are reactants, b is reagent and c is product
    return a list of pairs including [['src', 'targ'], ...]
    '''
    pairs = []
    for item in data:
        item = item.split('>')
        src = item[0] + '>' + item[1] # reactant and reagent
        targ = item[2]
        pairs.append([src, targ])
    
    return pairs

def filterPair(p, MAX_LENGTH):
    '''
    p stands for pair like ['src', 'targ']
    MAX_LENGTH means pair with length longer than MAX_LENGTH will be truncated to this fixed length
    return a list whose elements are truncated to MAX_LENGTH
    '''
    return len(p[0]) < MAX_LENGTH and len(p[1]) < MAX_LENGTH  # startswith first arg must be str or a tuple of str
