import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
    # Pass input through RNN layer
        _, hidden = self.rnn(inputs)

        # Get the output layer representation
        # output = self.W(rnn_output)
        output = self.W(hidden)        

        # Sum over the output along the sequence dimension (assuming it's the second dimension)
        output_sum = torch.sum(output, dim=1)

        # Apply log-softmax to get the predicted vector
        predicted_vector = self.softmax(output_sum)
        # print('--------------predicted_vector:', predicted_vector)
        return predicted_vector

def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        test = json.load(test_f)

    tra = []
    val = []
    tst = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in test:
        tst.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val, tst


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data)
    print("========== Vectorizing data ==========")
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    hidden_dims = [16, 32, 64,]
    results = []

    for dim in hidden_dims:
        print(f'======Starting dim: {dim}')
        stopping_condition = False
        epoch = 0

        last_train_accuracy = 0
        last_validation_accuracy = 0
        test_accuracy = 0
        
        # Initialize lists to store training loss and validation accuracy
        train_losses = []
        val_accuracies = []

        model = RNN(50, dim)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        while not stopping_condition:
            random.shuffle(train_data)
            model.train()

            print("Training started for epoch {}".format(epoch + 1))
            train_data = train_data
            correct = 0
            total = 0
            minibatch_size = 16
            N = len(train_data)

            loss_total = 0
            loss_count = 0
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                    input_words = " ".join(input_words)

                    # Remove punctuation
                    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                    # Look up word embedding dictionary
                    vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

                    vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                    output = model(vectors)

                    example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))

                    predicted_label = torch.argmax(output)

                    correct += int(predicted_label == gold_label)
                    total += 1
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss

                loss = loss / minibatch_size
                loss_total += loss.data
                loss_count += 1
                loss.backward()
                optimizer.step()
            print(loss_total / loss_count)
            train_losses.append(loss_total / loss_count)
            print("Training completed for epoch {}".format(epoch + 1))
            print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            training_accuracy = correct / total

            model.eval()
            correct = 0
            total = 0
            random.shuffle(valid_data)
            print("Validation started for epoch {}".format(epoch + 1))
            valid_data = valid_data

            for input_words, gold_label in tqdm(valid_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower in word_embedding.keys() else word_embedding['unk'] for i in input_words]

                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
            print("Validation completed for epoch {}".format(epoch + 1))
            print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            validation_accuracy = correct / total
            val_accuracies.append(validation_accuracy)

            if (epoch > 6 and validation_accuracy < last_validation_accuracy and training_accuracy > last_train_accuracy) or epoch > 11:
                stopping_condition = True
                print("Training done to avoid overfitting!")
                print("Best validation accuracy is:", last_validation_accuracy)
            else:
                last_validation_accuracy = validation_accuracy
                last_train_accuracy = training_accuracy

            epoch += 1

        correct = 0
        total = 0
        # random.shuffle(test_data)
        # print("test data processing")
        # test_data = test_data

        for input_words, gold_label in tqdm(test_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                    in input_words]

            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            # print(predicted_label, gold_label)
        print("Test data processing completed")
        print("Test accuracy: {}".format(correct / total))
        test_accuracy = correct/total
        
        results.append((dim, epoch, train_losses, val_accuracies, test_accuracy))


    # Summarize the results
    print("Hidden Dim | Epoch | Best Train Loss | Best Dev Accuracy | Test Accuracy")
    for hidden_dim, epoch, training_losses, validation_accs, test_accuracy in results:
        print(f"{hidden_dim:^10} | {epoch:^5} | {training_losses[-2]:^14.6f} | {validation_accs[-2]:^18.2%} | {test_accuracy:^15.2%}")

    for hidden_dim, epoch, training_losses, validation_accuracies, test_accuracy1 in results:
        print(f'{hidden_dim} : train: ', training_losses)
        print('validation: ', validation_accuracies)
        print('test', test_accuracy1)

    # best is before last overfit
    best_hidden_dim, epoch_of_best, best_training_losses, best_validation_accuracies, test_accuracy_of_best_model = max(results, key=lambda x: x[-2][-1])

    # Plot learning curve
    plt.figure(figsize=(12, 6))
    fig, ax1 = plt.subplots()
    # plt.subplot(1, 2, 1)
    ax2 = ax1.twinx()
    ax1.plot(range(1, epoch_of_best + 1), best_training_losses, 'b-', label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss", color = 'b')
    ax2.set_ylabel("Validation Accuracy", color='g')
    ax1.set_xticks(range(1, epoch_of_best + 1, 2))
    ax2.plot(range(1, epoch_of_best + 1), best_validation_accuracies, 'g-', label="Validation Accuracy")

    plt.title(f"RNN Analysis - Hidden Layer Dimension: {best_hidden_dim}")
    plt.tight_layout()
    plt.show()
    plt.savefig('rnn-val-fig.png')
