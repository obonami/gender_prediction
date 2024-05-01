import ast
import matplotlib.pyplot as plt
from typing import List
from itertools import cycle


# def plot_prediction_curve(word, predictions, true_class, binary=False):
#     """
#     If binary=True, plots the evolution of the true class probabilities over characters. 
#     Otherwise, plots the evolution of all class probabilities. 
#     """
#     class_names = list(predictions[0][1].keys())
#     if binary:
#         class_names = [true_class]  # cleaner to only plot the evolution of the true class
#     class_probs = [[tup[1][key] for tup in predictions] for key in class_names]
#     characters = [tup[0] for tup in predictions]
#     colors = {'m': 'steelblue', 'f': 'green', 'b': 'orange'}
#     plt.style.use('ggplot')
#     for i, class_i in enumerate(class_probs):
#         plt.plot(range(len(characters)), class_i, color=colors[class_names[i]], marker='x' , label=class_names[i])
#         for j, prob in enumerate(class_i):
#             plt.text(j, prob, f'{prob:.2f}', ha='center', va='bottom', fontsize=8)  # display probabilities as text
#     plt.title(f'Probability of each gender at each character position in "{word}"')
#     plt.xlabel('Character indecies')
#     plt.ylabel('Probability')
#     plt.xticks(range(len(characters)), characters)
#     plt.legend()
#     plt.show()


# def view_plateau(word, df, binary=False):
#     """
#     Checks to see if the word exists in the dataset and if so, plots the class probabilities at each character position for each run
#     """
#     if word in df['Form'].to_list():
#         probabilities = df[df['Form'] == word]['Class Probabilities'].apply(lambda x: ast.literal_eval(x)).tolist()
#         true_class = df[df['Form'] == word]['True Gender'].iloc[0]
#         for run in probabilities:	
#             plot_prediction_curve(word, run, true_class=true_class, binary=binary)
#     else:
#         print(f'{word} not found.')      


def plot_prediction_curve(words, words_data, binary=False):
    """
    If binary=True, plots the evolution of the true class probabilities over characters. 
    Otherwise, plots the evolution of all class probabilities. 
    """
    plt.style.use('ggplot')
    line_styles = ['-', ':', '--', '-.']
    line_colors = ['coral', 'darkslateblue', 'darkgray', 'teal', 'palevioletred', 'rteelblue', 'brown', 'gold']
    styles = [(style, color) for style, color in zip(cycle(line_styles), line_colors)]
    style = 0
    
    for word in words:
        word_predictions = words_data[words_data['Form'] == word]['Class Probabilities'].apply(lambda x: ast.literal_eval(x)).tolist()[0]
        true_class = words_data[words_data['Form'] == word]['True Gender'].item()
        
        if binary:
            class_names = [true_class]  # cleaner to only plot the evolution of the true class
        else:
            class_names = list(word_predictions[0][1].keys())

        class_probs = [[tup[1][key] for tup in word_predictions] for key in class_names]
        characters = [tup[0] for tup in word_predictions]   
        
        for i, class_i in enumerate(class_probs):
            line, = plt.plot(range(len(characters)), 
                             class_i, 
                             linestyle=styles[style % len(styles)][0], 
                             color=styles[style % len(styles)][1],
                             marker='x', 
                             label=f'{word} ({class_names[i]})')
            for j, prob in enumerate(class_i):
                plt.text(j, prob, f'{prob:.2f}', color=line.get_color(), ha='center', va='bottom', fontsize=8)  # to display the probabilities as text 
        style += 1
    
    if binary:
        plt.title('Probability of the true class at each character position')
    else:
        plt.title('Probability of each gender at each character position')
    plt.xlabel('Character indices')
    plt.ylabel('Probability')
    plt.legend()
    plt.show()


def view_plateau(words:List, df, binary=False, multiple_runs=False):
    for word in words:
        if word not in df['Form'].tolist():
            return f'"{word}" not found. Cannot proceed.'
            
    if multiple_runs:
        num_runs = df['Run'].nunique()
    else:
        num_runs = 1
    for run in range(1, num_runs + 1):
        print(f'Run {run} of {num_runs}:')
        try:
            words_data = df[(df['Form'].isin(words)) & (df['Run'] == run)]
        except KeyError:
            words_data = df[df['Form'].isin(words)]
        
        plot_prediction_curve(words, words_data, binary=binary)


def plot_metrics(train_acc, valid_acc, train_losses, valid_losses):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plt.subplots_adjust(hspace=0.5)

    n_epochs = range(1, len(valid_losses) + 1)

    # Plot accuracy
    ax1.set_title('Accuracy Evolution Over Epochs')
    ax1.plot(n_epochs, train_acc, marker='o', color='steelblue', label='Training Accuracy')
    ax1.plot(n_epochs, valid_acc, marker='o', color='orange', label='Validation Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, linestyle='--')

    # Plot loss
    ax2.set_title('Loss Evolution Over Epochs')
    ax2.plot(n_epochs, train_losses, marker='o', color='steelblue', label='Training Loss')
    ax2.plot(n_epochs, valid_losses, marker='o', color='orange', label='Validation Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, linestyle='--')

    plt.show()