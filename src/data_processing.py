import csv
from random import shuffle


def reverse_sequence(noun):
    return noun[::-1]


def pad_sequence(sequence, pad_size, pad_token):
    # returns a list of the characters in the sequence with additional pad tokens to match pad_size if needed
    return list(sequence) + [pad_token] * (pad_size - len(sequence))


def code_sequence(charseq, encodingmap, unk_token='<unk>'):
    # charseq is a sequence of chars
    return [encodingmap[char] if char in encodingmap 
            else encodingmap[unk_token] for char in charseq]


def decode_sequence(idxseq, decodingmap):
    # idxseq is a list of integers
    return [decodingmap[idx] for idx in idxseq]


def get_data(df, reverse_nouns=False):
    nouns = df.iloc[:,0].tolist()
    gender = df.iloc[:,1].tolist()   
    if reverse_nouns:
        nouns = [reverse_sequence(noun) for noun in nouns]
    noun_chars = [[char for char in noun] for noun in nouns]
    return noun_chars, gender


def vocabulary(df, labels=False, pad_token='<pad>', unk_token='<unk>'):

    nouns, genders = get_data(df, reverse_nouns=False)
    
    if labels:
        sym2idx = {sym: idx for idx, sym in enumerate(set(genders))}
    else:
        unique_chars = set(char for noun in nouns for char in noun)
        sym2idx = {sym: idx for idx, sym in enumerate(unique_chars)}
        sym2idx[unk_token] = len(sym2idx)
        sym2idx[pad_token] = len(sym2idx)

    idx2sym = [sym for sym in sym2idx.keys()]

    return idx2sym, sym2idx


def save_padded_words(filename, batch_of_words):
    lines = ['\t'.join(word) + '\n' for word in batch_of_words]
    with open(filename, 'a', encoding='utf-8') as f:
        f.writelines(lines)


def save_probabilities(probabilities, df, filename, mode, set):
    """
    Args:
        probabilities: dict showing the probability of each class at each character position.
        df: Pandas DataFrame object containing the true gender for each word
        mode: 'w' (to overwrite the file) or 'a' (to append to the file)
        set: a str showing which set the word belongs to ('Train' / 'Validation' / 'Test') 
        filename: the name of a csv file to write the results to
    """
    # Sorting the words in alphabetical order
    sorted_items = dict(sorted(probabilities.items()))

    # Dictionary mapping words to their true genders
    word_to_gender = dict(zip(df.iloc[:,0], df['gen']))

    assert mode in ['w', 'a'], "The mode needs to be either 'w' (to overwrite the file) or 'a' (to append to the file)"

    with open(filename, mode) as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(['Nouns', 'Class Probabilities', 'True Gender', 'Set'])
        for word, pred_probs in sorted_items.items():
            true_gender = word_to_gender.get(word, 'Gender not found')
            writer.writerow([word, pred_probs, true_gender, set])

        print(f'File successfully written to {filename}.')


class DataGenerator:

      def __init__(self, df, parentgenerator=None, reverse_nouns=False, pad_token='<pad>', unk_token='<unk>'):

            if parentgenerator is not None: # Reuses the encodings of the parent if specified
                self.pad_token      = parentgenerator.pad_token
                self.unk_token      = parentgenerator.unk_token
                self.input_sym2idx  = parentgenerator.input_sym2idx
                self.input_idx2sym  = parentgenerator.input_idx2sym
                self.output_sym2idx = parentgenerator.output_sym2idx
                self.output_idx2sym = parentgenerator.output_idx2sym
            else:                           # Creates new encodings
                self.pad_token = pad_token
                self.unk_token = unk_token
                self.input_idx2sym, self.input_sym2idx   = vocabulary(df, labels=False)
                self.output_idx2sym, self.output_sym2idx = vocabulary(df, labels=True)

            nouns, genders = get_data(df, reverse_nouns=reverse_nouns)
            self.X = nouns
            self.Y = genders


      def generate_batches(self, batch_size):

            assert(len(self.X) == len(self.Y))

            N     = len(self.X)
            idxes = list(range(N))

            # data ordering
            shuffle(idxes)
            idxes.sort(key=lambda idx: len(self.X[idx]))

            # batch generation
            bstart = 0
            while bstart < N:
                bend        = min(bstart + batch_size, N)
                batch_idxes = idxes[bstart:bend]
                batch_len   = max(len(self.X[idx]) for idx in batch_idxes)

                padded_X = [pad_sequence(self.X[idx], batch_len, self.pad_token) for idx in batch_idxes]
                #   save_padded_words('../data/eval/padded_nouns', padded_X)
                seqX = [code_sequence(seq, self.input_sym2idx, self.unk_token) for seq in padded_X]
                seqY = [self.output_sym2idx[self.Y[idx]] for idx in batch_idxes]

                assert(len(seqX) == len(seqY))
                yield (seqX, seqY)
                bstart += batch_size