import matplotlib.pyplot as plt

from data_processing import DataGenerator
from model import GenderLSTM


def baseline_accuracy(traingenerator, validgenerator, verbose=False):
    _, train_labels = list(*traingenerator.generate_batches(len(traingenerator.X)))
    most_frequent_label = max(set(train_labels), key=train_labels.count)   
    
    if verbose:
        print(f'The most frequent label in the dataset is: {traingenerator.output_idx2sym[most_frequent_label]}')

    _, dev_labels = list(*validgenerator.generate_batches(len(validgenerator.X)))
    return dev_labels.count(most_frequent_label) / len(dev_labels)


def compare_accuracies(baseline_acc, model_acc):
    
    x_labels = ['Baseline', 'Model']
    bar_colors = ['#FFDFD3', '#593e67'] # '#957DAD'
    bars = plt.bar(x_labels, [baseline_acc, model_acc], width=0.3, color=bar_colors)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', ha='center', va='bottom')

    plt.title('Comparison Between Model Accuracy and Baseline Accuracy (= Most Frequent Class)')
    plt.ylabel('Accuracy')
    plt.grid(linestyle='dashed')
    ax = plt.gca()
    ax.set_axisbelow(True)

    plt.show()


def statistical_check(trainset, validset, hyperparameters, runs=10, reverse_nouns=True, device='cpu'):
    """
    Returns the accuracy, loss, plateau beginning index, and accuracy at plateau beginning index 
            averaged over a specified number of runs for both training and validation sets.
    """
    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    train_idx_plateau_beg, train_acc_plateau_beg, valid_idx_plateau_beg, valid_acc_plateau_beg = [], [], [], []
    
    embedding_dim = hyperparameters['embed_dim']
    hidden_size = hyperparameters['hidden_size']
    batch_size = hyperparameters['batch_size']
    n_epochs = hyperparameters['n_epochs']
    lr = hyperparameters['lr']

    for _ in range(runs):
        train_generator = DataGenerator(trainset, reverse_nouns=reverse_nouns)
        valid_generator = DataGenerator(validset, reverse_nouns=reverse_nouns)
        model = GenderLSTM(train_generator, embedding_dim, hidden_size, device=device)
        train, val = model.train_model(train_generator, valid_generator, n_epochs, batch_size, lr, verbose=False, save_model=False)
        
        train_loss.append(train['loss'][-1])
        train_acc.append(train['accuracy'][-1])
        valid_loss.append(val['loss'][-1])
        valid_acc.append(val['accuracy'][-1])

        train_idx_plateau_beg.append(train['plateau_beg'][-1])
        train_acc_plateau_beg.append(train['acc_at_plateau_beg'][-1])
        valid_idx_plateau_beg.append(val['plateau_beg'][-1])
        valid_acc_plateau_beg.append(val['acc_at_plateau_beg'][-1])

    t = {'avg_accuracy': sum(train_acc) / len(train_acc),
         'avg_loss': sum(train_loss) / len(train_loss),
         'avg_plateau_beg': int(sum(train_idx_plateau_beg) / len(train_idx_plateau_beg)), 
         'avg_acc_at_plateau_beg': sum(train_acc_plateau_beg) / len(train_acc_plateau_beg)
        }
            
    v = {'avg_accuracy': sum(valid_acc) / len(valid_acc),
         'avg_loss': sum(valid_loss) / len(valid_loss),
         'avg_plateau_beg': int(sum(valid_idx_plateau_beg) / len(valid_idx_plateau_beg)), 
         'avg_acc_at_plateau_beg': sum(valid_acc_plateau_beg) / len(valid_acc_plateau_beg)
        }

    print(f"Avg Train Loss: {t['avg_loss']:.4f}   Avg Train Accuracy: {t['avg_accuracy'] * 100:.2f}%   Avg Plateau Beginning (index): {t['avg_plateau_beg']}   Avg Plateau Beginning Accuracy: {t['avg_acc_at_plateau_beg'] * 100:.2f}%")      
    print(f"Avg Valid Loss: {v['avg_loss']:.4f}   Avg Valid Accuracy: {v['avg_accuracy'] * 100:.2f}%   Avg Plateau Beginning (index): {v['avg_plateau_beg']}   Avg Plateau Beginning Accuracy: {v['avg_acc_at_plateau_beg'] * 100:.2f}%")

    return t, v