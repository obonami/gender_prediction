import csv
from random import shuffle

import torch
from sklearn.model_selection import train_test_split


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


def get_data_from_df(df, reverse_nouns=False):
    nouns = df.iloc[:,0].tolist()
    gender = df.iloc[:,1].tolist()   
    if reverse_nouns:
        nouns = [reverse_sequence(noun) for noun in nouns]
    noun_chars = [[char for char in noun] for noun in nouns]
    return noun_chars, gender


def vocabulary(df, labels=False, pad_token='<pad>', unk_token='<unk>'):

    nouns, genders = get_data_from_df(df, reverse_nouns=False)
    
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


def save_probabilities(model_checkpoint, df, filename):
    checkpoint = torch.load(model_checkpoint)
    train = checkpoint['train_char_prediction_probs']
    valid = checkpoint['valid_char_prediction_probs']

    # Sorting the words in alphabetical order
    sorted_train = dict(sorted(train.items()))
    sorted_valid = dict(sorted(valid.items()))

    # Dictionary mapping words to their true genders
    word_to_gender = dict(zip(df.iloc[:,0], df['gender']))

    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Nouns', 'Class Probabilities', 'True Gender', 'Set'])
        for valid_word, valid_pred_probs in sorted_valid.items():
            true_gender = word_to_gender.get(valid_word, 'Gender not found')
            writer.writerow([valid_word, valid_pred_probs, true_gender, 'Validation'])

        for train_word, train_pred_probs in sorted_train.items():
            true_gender = word_to_gender.get(train_word, 'Gender not found')
            writer.writerow([train_word, train_pred_probs, true_gender, 'Training'])
        print(f'File successfully written to {filename}.')


class DataGenerator:

      def __init__(self, data, reverse_nouns=False, pad_token='<pad>', unk_token='<unk>'):

            self.pad_token = pad_token
            self.unk_token = unk_token

            self.input_idx2sym,self.input_sym2idx   = vocabulary(data,False)
            self.output_idx2sym,self.output_sym2idx = vocabulary(data,True)

            nouns, genders = get_data_from_df(data, reverse_nouns=reverse_nouns)
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(nouns, genders, test_size=0.2)

            self.train_size = len(self.X_train)
            self.test_size = len(self.X_test)

      def generate_batches(self,batch_size,validation=False):

            if validation:
                X = self.X_test
                y = self.y_test
            else:
                X = self.X_train
                y = self.y_train

            assert(len(X) == len(y))

            N     = len(X)
            idxes = list(range(N))

            # data ordering
            shuffle(idxes)
            idxes.sort(key=lambda idx: len(X[idx]))

            # batch generation
            bstart = 0
            while bstart < N:
                bend        = min(bstart+batch_size,N)
                batch_idxes = idxes[bstart:bend]
                batch_len   = max(len(X[idx]) for idx in batch_idxes)
                Xpad        = [pad_sequence(X[idx],batch_len,self.pad_token) for idx in batch_idxes]
                #   save_padded_words('../data/eval/padded_fr', Xpad)
                seqX        = [code_sequence(x,self.input_sym2idx,self.unk_token) for x in Xpad]
                seqY        = [self.output_sym2idx[y[idx]] for idx in batch_idxes]

                assert(len(seqX) == len(seqY))
                yield (seqX,seqY)
                bstart += batch_size