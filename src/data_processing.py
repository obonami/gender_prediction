import csv
import pandas as pd
from random import shuffle


def reverse_sequence(noun):
    return str(noun)[::-1]


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



def get_correct_wrong_pred_df(pred_df, pred_col, proportions):
    
    dfs = []
    runs = pred_df['Run'].unique()
    for run in runs:
        run_data = pred_df[pred_df['Run'] == run]
        crosstab = pd.crosstab(run_data[pred_col], run_data['true'])
        
        # Extract counts for true and false predictions for each gender
        f_true = crosstab.loc['f', 'f'] if 'f' in crosstab.index else 0
        m_true = crosstab.loc['m', 'm'] if 'm' in crosstab.index else 0
        f_false = crosstab.loc['f', 'm'] if 'm' in crosstab.index else 0
        m_false = crosstab.loc['m', 'f'] if 'f' in crosstab.index else 0
        
        if proportions :
            total_f = f_true + f_false
            total_m = m_true + m_false
            f_true = round(f_true / total_f, 3) if total_f > 0 else 0
            m_true = round(m_true / total_m, 3) if total_m > 0 else 0
            f_false = round(f_false / total_f, 3) if total_f > 0 else 0
            m_false = round(m_false / total_m, 3) if total_m > 0 else 0
        
        run_dict = {
            'Run': run,
            'f_true': f_true,
            'm_true': m_true,
            'f_false': f_false,
            'm_false': m_false
        }
        
        dfs.append(run_dict)

    return pd.DataFrame(dfs)



def get_category_gender_partition(category, echantinom, pred_df, pred_col, run=None, proportion=False):
    # TODO: there is an issue with the count
    if run is not None:
        pred_df = pred_df[pred_df['Run'] == run]

    true_cross_tab = pd.crosstab(echantinom[echantinom['lemma'].isin(pred_df['lemma'])][category], pred_df['true'])
    true_cross_tab.columns = ['f_true', 'm_true']

    f_false = pred_df[(pred_df[pred_col] == 'f') & (pred_df['true'] == 'm')].groupby(echantinom[category]).size().rename('f_false')
    m_false = pred_df[(pred_df[pred_col] == 'm') & (pred_df['true'] == 'f')].groupby(echantinom[category]).size().rename('m_false')

    combined_df = pd.concat([true_cross_tab, f_false, m_false], axis=1)
    combined_df.fillna(0, inplace=True)
    combined_df = combined_df.loc[combined_df.sum(axis=1).sort_values(ascending=False).index]

    if proportion:
        f_total = combined_df['f_true'] + combined_df['f_false']
        m_total = combined_df['m_true'] + combined_df['m_false']
        combined_df['f_true'] = round(combined_df['f_true'] / f_total, 3)
        combined_df['f_false'] = round(combined_df['f_false'] / f_total, 3)
        combined_df['m_true'] = round(combined_df['m_true'] / m_total, 3)
        combined_df['m_false'] = round(combined_df['m_false'] / m_total, 3)

    return combined_df




def get_false_preds(run, echantinom, pred_col, pred_gender, true_gender, pred_df, category, subcategory):
   
    f_false_rows = pred_df[(pred_df['Run'] == run) & (pred_df[pred_col] == pred_gender) & (pred_df['true'] == true_gender)]

    # Merge to get the 'category' column
    f_false_rows = f_false_rows.merge(echantinom[['lemma', category]], how='left', left_on='lemma', right_on='lemma')

    # Filter to keep only the subcategory rows 
    simplex_f_false_rows = f_false_rows[f_false_rows[category] == subcategory]
    return simplex_f_false_rows



def get_subcategories_count_per_run(pred_df, pred_gender, true_gender, category, echantinom, col= 'orth_pred'):
    all_runs = []
    for run in range(10):
        f_false_rows = pred_df[(pred_df['Run'] == run) & (pred_df[col] == pred_gender) & (pred_df['true'] == true_gender)]

        f_false_rows = f_false_rows.merge(echantinom[['lemma', category]], how='left', on='lemma')

        run_counts = f_false_rows.groupby(category)['lemma'].count().reset_index()
        run_counts['Run'] = run  
        all_runs.append(run_counts)

    all_runs_df = pd.concat(all_runs, ignore_index=True)
    pivot_table = all_runs_df.pivot_table(index=category, columns='Run', values='lemma', fill_value=0)
    return pivot_table




def most_common(series):
    return series.value_counts().index[0]