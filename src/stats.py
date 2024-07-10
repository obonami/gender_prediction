""" The lower the EMD cost, the more similar the two distributions are to each other. """ 

import ast
from scipy.stats import wasserstein_distance

class EMD:

    def __init__(self, df):
        self.df = df


    def cost(self, word_a, word_b, run=None):
        probs_a = self._word_run_probs(word_a, run)
        probs_b = self._word_run_probs(word_b, run)
        
        if len(probs_a) != len(probs_b):
            min_length = min(len(probs_a), len(probs_b))
            probs_a = probs_a[:min_length]
            probs_b = probs_b[:min_length]
        
        return wasserstein_distance(probs_a, probs_b)
    

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