import ast
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance


class Distance:

    def __init__(self, df):
        self.df = df

    """ Earth Mover's (Wasserstein) Distance """
    """ Lower EMD values indicate more similar distributions """
    def EMD(self, word_a, word_b, run=None):
        probs_a, probs_b = self._get_probs(word_a, word_b, run)        
        return wasserstein_distance(probs_a, probs_b)
    

    """ Euclidean Distance """
    """ Lower Euclidean distances indicate more similar sequences """
    def euclidean(self, word_a, word_b, run=None):
        probs_a, probs_b = self._get_probs(word_a, word_b, run)    
        return euclidean(probs_a, probs_b)
    

    def _get_probs(self, word_a, word_b, run):
        probs_a = self._word_run_probs(word_a, run)
        probs_b = self._word_run_probs(word_b, run)

        if len(probs_a) != len(probs_b):
            probs_a, probs_b = self._clip_length(probs_a, probs_b)
        
        return probs_a, probs_b
    
    
    def _word_run_probs(self, word, run):
        filtered_df = self.df[self.df['Form'] == word]
        if run:
            filtered_df = filtered_df[filtered_df['Run'] == run]

        if filtered_df.empty:
            raise ValueError(f'No data found for word "{word}".') 
               
        row = filtered_df.iloc[0]
        return self._true_class_probs(row['Class Probabilities'], row['True Gender'])
    
    
    def _true_class_probs(self, class_probabilities, true_gender):
        class_probabilities = ast.literal_eval(class_probabilities)
        return [prob_dict[true_gender] for _, prob_dict in class_probabilities]

    
    def _clip_length(self, seq1, seq2):
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]
        return seq1, seq2