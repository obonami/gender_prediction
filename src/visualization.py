import ast
import matplotlib.pyplot as plt


def plot_prediction_curve(word, predictions):
    class_names = list(predictions[0][1].keys())
    class_probs = [[tup[1][key] for tup in predictions] for key in class_names]
    characters = [tup[0] for tup in predictions]
    for i, class_i in enumerate(class_probs):
        plt.plot(range(len(characters)), class_i, label=class_names[i])
    plt.title(f'Probability of each gender at each character position in "{word}"')
    plt.xlabel('Character indecies')
    plt.ylabel('Probability')
    plt.xticks(range(len(characters)), characters)
    plt.legend()
    plt.show()
    print(f'Probability values:\n  {predictions}')


def view_plateau(word, df):
    """
    Checks to see if the word exists in the dataset and if so, plots the accuracy at each character position
    """
    if word in df['Word'].to_list():
        probabilities = df[df['Word'] == word]['Class Probabilities'].apply(lambda x: ast.literal_eval(x)).tolist()[0]
        plot_prediction_curve(word, probabilities)

    else:
        print(f'{word} not found.')


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