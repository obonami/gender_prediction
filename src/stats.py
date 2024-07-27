import ast
from typing import List, Tuple, Dict, Union

import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
import seaborn as sns


class Distance:

    def __init__(self, df):
        self.df = df

    """ Earth Mover's (Wasserstein) Distance """
    """ Lower EMD values indicate more similar distributions """
    def EMD(self, seq1, seq2, run=None):
        probs1, probs2 = self._get_probs(seq1, seq2, run)        
        return wasserstein_distance(probs1, probs2)
    

    """ Euclidean Distance """
    """ Lower Euclidean distances indicate more similar sequences """
    def euclidean(self, seq1, seq2, run=None):
        probs1, probs2 = self._get_probs(seq1, seq2, run)    
        return euclidean(probs1, probs2)
    

    def _get_probs(self, seq1, seq2, run):
        probs1 = self._word_run_probs(seq1, run) if isinstance(seq1, str) else seq1
        probs2 = self._word_run_probs(seq2, run) if isinstance(seq2, str) else seq2

        if len(probs1) != len(probs2):
            probs1, probs2 = self._clip_length(probs1, probs2)
        
        return probs1, probs2
    
    
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
    


class SuffixAverage:

    def __init__(self, df, suffix:str):
        self.suffix = suffix
        self.df = df[df['suffix'] == suffix].copy()


    def prob(self, gender:str, min_dp:int=0):
        assert gender in ['True', 'f', 'm'], "Possible values for the gender parameter are 'True', 'f', or 'm'."      
        df = self._prepare_df(self.df, gender, min_dp)
        return df.groupby('Position')['Probs'].mean().tolist()


    def plot(self, gender:str, min_dp:int=0, title=True, scale:Union[bool,List[int]]=False) -> None:
        """
        Plots the average probability at each character position for words with a given suffix

        gender: Possible values: 'True', 'f', 'm'
        min_dp: minimum number of data points required for a position to be considered
        scale:  Whether or not to scale the y_axis to a certain range.
                Possible values: 
                    - True: Scales from 0 to 1 
                    - False: The range will be from the minimum to maximum value (default)
                    - List[int]: Custom scaling range
        """
        assert gender in ['True', 'f', 'm'], "Possible values for the gender parameter are 'True', 'f', or 'm'."
        
        df = self._prepare_df(self.df, gender, min_dp)
        
        sns.set_theme(style="darkgrid")
        plot = sns.relplot(data=df, x='Position', y='Probs', kind='line')
        
        gender_mapping = {'m': 'masculine', 'f': 'feminine'}
        plot.set_axis_labels("Character Position (from the end)", 
                            f"Probability of {gender_mapping.get(gender, 'True gender')}")

        if scale is True:
            plt.ylim(0, 1)
        elif isinstance(scale, list) and len(scale) == 2:
            plt.ylim(scale[0], scale[1])
        # Otherwise, the y-axis will be automatically scaled from min to max values

        # Ensure x-axis displays integer values starting from 1
        ax = plt.gca()
        positions = df['Position'].unique()
        ax.set_xticks(positions)
        ax.set_xticklabels(positions + 1)

        if title:
            n_samples = self.df.Form.nunique()
            plot.figure.suptitle(
                f'Average Probability Distribution for Words with Suffix: "{self.suffix}" ({n_samples} samples)', 
                size=14, x=0.56
                )
        plot.tight_layout()
        plt.show()


    def _prepare_df(self, df, gender:str, min_dp:int=0):
        df['Probs'] = df.apply(lambda row: self._extract_class_probs(
            row['Class Probabilities'], row['True Gender'] if gender == 'True' else gender
            ), axis=1)

        # Separating each probability into its own row
        df = df.explode('Probs')
        
        # Keeping track of the position of each probability within the original list before the explode operation
        df['Position'] = df.groupby(level=0).cumcount()

        # Filtering out positions that appear less than min_dp times
        df = df[df.groupby('Position')['Position'].transform('size') >= min_dp]

        return df
    

    def _extract_class_probs(self, word_probs: List[Tuple[str, Dict[str, float]]], gen:str) -> List[float]:
        try:
            result = [tup[1][gen] for tup in ast.literal_eval(word_probs)]
            return result
        except Exception as e:
            print(f"Error processing {word_probs} with gender {gen} ({e}). Maybe an incorrect suffix is selected?")
            return []