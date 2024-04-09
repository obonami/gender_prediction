import statistics

import torch
import torch.nn as nn
import torch.optim as optim

from data_processing import reverse_sequence


class GenderLSTM(nn.Module):

    def __init__(self, datagenerator, embedding_dim, hidden_dim, device='cpu', reversed=False):
        super(GenderLSTM, self).__init__()
        self.to(device)
        self.reversed = reversed

        invocab_size = len(datagenerator.input_idx2sym)
        outvocab_size = len(datagenerator.output_idx2sym)

        self.device = torch.device(device)
        self.pad_idx = datagenerator.input_sym2idx[datagenerator.pad_token]
        self.embedding = nn.Embedding(invocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, outvocab_size)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, inputs):
        embeds = self.embedding(inputs)
        
        # lstm_out contains the hidden states for each time step in the input sequence for each element in the batch. Shape: [batch_size, sequence_length, hidden_size]
        # hidden is the hidden state for the last time step in the input sequence for each element in the batch. Shape: [num_layers(* num_directions which is 1 here), batch_size, hidden_size]
        lstm_out, (hidden, _) = self.lstm(embeds)   
        
        out_logits = self.fc(lstm_out)   # shape: [batch_size, sequence_length, num_classes] --> these will be used to get accuracy values for each time step
        
        #  taking the last hidden state of the last layer with hidden[-1]
        hidden_logits = self.fc(hidden[-1]) # shape: [batch_size, num_classes]  --> these will be used for loss computation

        # add softmax to get probability distributions (for easier interpretability)
        out_probabilities = self.softmax(out_logits)   # probability distribution over the classes for each character in the sequence 
        # it would also be a good idea to calibrate the model (dividing c by a value T (tempreture) that is a hyperparameter that can be tuned --> ask Timothee)
        return out_probabilities, hidden_logits  


    def train_model(self, traingenerator, validgenerator, n_epochs, batch_size, learning_rate=0.001, verbose=True, save_model=False):
        """
        save_model must be either False or a string showing the path where the model should be saved.
        """
        criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        train = {'accuracy': [],
                 'loss': [],
                 }
            
        valid = {'accuracy': [],
                 'loss': [],
                 }

        min_val_loss = float('inf')
        patience = 3
        counter = 0
        
        for epoch in range(n_epochs):
            
            if verbose:
                print(f'Epoch: {epoch}')
            
            epoch_train_accuracies = [] 
            epoch_valid_accuracies = []
            epoch_train_losses = []
            epoch_valid_losses = []
            train_plateau_indecies = [] # will contain the index of the position at which the accuracy starts to plateau for each batch
            valid_plateau_indecies = []
            train_plateau_accuracies = []   # will contain the accuracy of the position at which the accuracy starts to plateau for each batch
            valid_plateau_accuracies = []

            train_char_prediction_probs = {} # will contain the tokens in the training set as keys and lists of 0s or 1s indicating incorrect or correct prediction at each character position in the tokens as values
            valid_char_prediction_probs = {} # will contain the tokens in the validation set as keys and lists of 0s or 1s indicating incorrect or correct prediction at each character position in the tokens as values
            
            # Training phase
            self.train()
            for inputs, labels in traingenerator.generate_batches(batch_size):
                X = torch.LongTensor(inputs).to(self.device) # shape: [batch_size, sequence_length]
                Y = torch.LongTensor(labels).to(self.device) # shape: [batch_size]

                optimizer.zero_grad()
                Y_seq_probs, Y_hidden_logits = self.forward(X) # shapes: [batch_size, sequence_length, num_classes], [batch_size, num_classes]

                # Loss
                loss = criterion(Y_hidden_logits, Y)
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())

                # Accuracy
                Y_pred = torch.argmax(Y_hidden_logits, dim=-1)  # shape: [batch_size]
                correct = (Y_pred == Y)
                avg_batch_accuracy = correct.float().mean()  # scalar
                epoch_train_accuracies.append(avg_batch_accuracy)

                # Character-level predictions
                pad_idx = traingenerator.input_sym2idx[traingenerator.pad_token]
                num_classes = Y_seq_probs.size(-1)          
                for word_idx in range(len(X)):
                    word = ''.join(traingenerator.input_idx2sym[char_id] for char_id in X[word_idx] if char_id != pad_idx)
                    # If the words are reversed, return them to their normal form
                    if self.reversed:
                        word = reverse_sequence(word)
                    train_char_prediction_probs[word] = [
                        (
                            traingenerator.input_idx2sym[char_id], 
                            {
                                traingenerator.output_idx2sym[class_idx]: Y_seq_probs[word_idx, char_idx, class_idx].item() 
                                for class_idx in range(num_classes)
                            }
                        )
                        for char_idx, char_id in enumerate(X[word_idx]) if char_id != pad_idx
                        ]

                # Plateau info
                plateau_beginning, plateau_confidence = self.get_plateau_info(Y_seq_probs, threshold=0.1, patience=3) 
                train_plateau_indecies.append(plateau_beginning)
                train_plateau_accuracies.append(plateau_confidence)


            # Validation phase
            self.eval()
            for val_inputs, val_labels in validgenerator.generate_batches(batch_size):
                with torch.no_grad():
                    X = torch.LongTensor(val_inputs).to(self.device)
                    Y = torch.LongTensor(val_labels).to(self.device)

                    Y_seq_probs, Y_hidden_logits = self.forward(X)

                    # Loss
                    loss = criterion(Y_hidden_logits, Y)
                    epoch_valid_losses.append(loss.item())

                    # Accuracy
                    Y_pred = torch.argmax(Y_hidden_logits, dim=-1)  # shape: [batch_size]
                    correct = (Y_pred == Y)
                    avg_batch_accuracy = correct.float().mean()  # scalar
                    epoch_valid_accuracies.append(avg_batch_accuracy)

                    # Character-level predictions         
                    for word_idx in range(len(X)):
                        word = ''.join(validgenerator.input_idx2sym[char_id] for char_id in X[word_idx] if char_id != pad_idx)
                        # If the words are reversed, return them to their normal form
                        if self.reversed:
                            word = reverse_sequence(word)
                        valid_char_prediction_probs[word] = [
                            (
                                validgenerator.input_idx2sym[char_id], 
                                {
                                    validgenerator.output_idx2sym[class_idx]: Y_seq_probs[word_idx, char_idx, class_idx].item() 
                                    for class_idx in range(num_classes)
                                }
                            )
                            for char_idx, char_id in enumerate(X[word_idx]) if char_id != pad_idx
                            ]

                    # Plateau info
                    plateau_beginning, plateau_confidence = self.get_plateau_info(Y_seq_probs, threshold=0.1, patience=3) 
                    valid_plateau_indecies.append(plateau_beginning)
                    valid_plateau_accuracies.append(plateau_confidence) 


            # store (and, if verbose is True, report) metrics at the end of each epoch
            avg_train_accuracy = sum(epoch_train_accuracies) / len(epoch_train_accuracies)
            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)  
            train_plateau_beg_mode = statistics.mode(train_plateau_indecies)
            train_avg_acc_at_mode_plateau_beg = statistics.mean(
                [train_plateau_accuracies[idx].item() for idx, value in enumerate(train_plateau_indecies) if value == train_plateau_beg_mode])                        
            
            avg_valid_accuracy = sum(epoch_valid_accuracies) / len(epoch_valid_accuracies)
            avg_valid_loss = sum(epoch_valid_losses) / len(epoch_valid_losses)
            valid_plateau_beg_mode = statistics.mode(valid_plateau_indecies)
            valid_avg_acc_at_mode_plateau_beg = statistics.mean(
                [valid_plateau_accuracies[idx].item() for idx, value in enumerate(valid_plateau_indecies) if value == valid_plateau_beg_mode])

            train['accuracy'].append(avg_train_accuracy)
            train['loss'].append(avg_train_loss)
            
            valid['accuracy'].append(avg_valid_accuracy)
            valid['loss'].append(avg_valid_loss)

            if verbose:
                print(f'[Train] Loss: {avg_train_loss:.4f}   Accuracy: {avg_train_accuracy * 100:.2f}%   Beginning of plateau (index): {train_plateau_beg_mode}   Accuracy at the beginning of plateau: {train_avg_acc_at_mode_plateau_beg * 100:.2f}%')      
                print(f'[Valid] Loss: {avg_valid_loss:.4f}   Accuracy: {avg_valid_accuracy * 100:.2f}%   Beginning of plateau (index): {valid_plateau_beg_mode}   Accuracy at the beginning of plateau: {valid_avg_acc_at_mode_plateau_beg * 100:.2f}%')
                print('-' * 100)

            # checking for early stopping
            if avg_valid_loss < min_val_loss:
                min_val_loss = avg_valid_loss
                counter = 0
                if save_model:
                    # save model checkpoint
                    checkpoint = {
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'train_accuracy': avg_train_accuracy,
                        'train_loss': avg_train_loss,
                        'train_plateau_beg': train_plateau_beg_mode,
                        'train_acc_at_plateau_beg': train_avg_acc_at_mode_plateau_beg,
                        'train_char_prediction_probs': train_char_prediction_probs,
                        'valid_accuracy': avg_valid_accuracy,
                        'valid_loss': avg_valid_loss,
                        'valid_plateau_beg': valid_plateau_beg_mode,
                        'valid_acc_at_plateau_beg': valid_avg_acc_at_mode_plateau_beg,
                        'valid_char_prediction_probs': valid_char_prediction_probs,
                    }
                    torch.save(checkpoint, save_model)
            else:
                counter += 1

            if counter == patience:
                if verbose:
                    print(f'Early stopping after {epoch + 1} epochs and {patience} epochs without improvement.')
                break

        return train, valid


    def predict(self, datagenerator, batch_size):
        predictions = {'Word': [], 'Predicted Gender': [], 'True Gender': []}
        
        self.eval()
        for inputs, labels in datagenerator.generate_batches(batch_size):
            with torch.no_grad():
                X = torch.LongTensor(inputs).to(self.device)
                Y = torch.LongTensor(labels).to(self.device)

                _, logits = self.forward(X)
                Y_pred = torch.argmax(logits, dim=-1)

                # converting the predictions into readable format
                pad_idx = datagenerator.input_sym2idx[datagenerator.pad_token]
                for i in range(len(X)):
                    predictions['Word'].append(''.join([datagenerator.input_idx2sym[idx] for idx in X[i] if idx != pad_idx]))
                    predictions['Predicted Gender'].append(datagenerator.output_idx2sym[Y_pred[i]])
                    predictions['True Gender'].append(datagenerator.output_idx2sym[Y[i]])
        
        return predictions


    def get_plateau_info(self, Y_probs, threshold=0.01, patience=3):
        prediction_probs, _ = torch.max(Y_probs, dim=-1)  # shape: [batch_size, sequence_length]
        avg_pos_confidence = prediction_probs.mean(dim=0) # shows the average confidence in prediction for each character position regardless of whether the prediction is correct or not

        # self.plot_confidence_curve(avg_pos_confidence.tolist())
        
        # if the confidence level doesn't increase by a value higher than the threshold for patience consecutive characters, extract the index and accuracy of the initial position
        counter = 0
        for position, confidence in enumerate(avg_pos_confidence):
            if position == 0: 
                continue
            elif confidence - avg_pos_confidence[position - 1] >= threshold:
                counter = 0
            elif confidence - avg_pos_confidence[position - 1] < threshold:
                counter += 1

            if counter == patience:
                plateau_beginning = position - patience
                plateau_confidence = avg_pos_confidence[position - patience]
                return plateau_beginning, plateau_confidence

        # if there is no plateau, returns the position and accuracy of the last character
        return len(avg_pos_confidence)-1, avg_pos_confidence[-1]
    
    
    # def plot_confidence_curve(self, probabilities):
    #     plt.plot(range(len(probabilities)), probabilities, marker='o')
    #     plt.title('Average Prediction Confidence at Each Character Position')
    #     plt.xlabel('Character indecies')
    #     plt.ylabel('Confidence Level')
    #     plt.show()