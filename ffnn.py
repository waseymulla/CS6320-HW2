import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden_rep = self.activation(self.W1(input_vector))
        # [to fill] obtain output layer representation
        output_rep = self.W2(hidden_rep)
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output_rep)
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding fito the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val

def train_and_evaluate_model(hidden_dim, epochs, train_data, valid_data):
    model = FFNN(input_dim=len(vocab), h=hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print(f"========== Training for {epochs} epochs with hidden_dim={hidden_dim} ==========")

    training_losses = []  # Store training loss for each epoch
    validation_accuracies = []  # Store validation accuracy for each epoch

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print(f"Training started for epoch {epoch + 1}")
        random.shuffle(train_data)  # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(train_data)

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()

        # Record the training loss for this epoch
        training_losses.append(loss.item())

        print(f"Training completed for epoch {epoch + 1}")
        print(f"Training accuracy for epoch {epoch + 1}: {correct / total:.2%}")
        print(f"Training time for this epoch: {time.time() - start_time}")

        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print(f"Validation started for epoch {epoch + 1}")
        minibatch_size = 16
        N = len(valid_data)

        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size

        # Record the validation accuracy for this epoch
        validation_accuracy = correct / total
        validation_accuracies.append(validation_accuracy)

        print(f"Validation completed for epoch {epoch + 1}")
        print(f"Validation accuracy for epoch {epoch + 1}: {validation_accuracy:.2%}")
        print(f"Validation time for this epoch: {time.time() - start_time}")

    return training_losses, validation_accuracies


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    
    hidden_dims = [32, 64, 128]  # You can change these hidden unit sizes
    results = []

    for hidden_dim in hidden_dims:
        training_losses, validation_accuracies = train_and_evaluate_model(hidden_dim, args.epochs, train_data, valid_data)
        results.append((hidden_dim, training_losses, validation_accuracies))



        # Summarize the results
    print("Hidden Dim | Best Train Loss | Best Dev Accuracy")
    best_dev_accuracy = 0
    best_hidden_dim = 0
    for hidden_dim, training_losses, validation_accuracies in results:
        best_train_loss = min(training_losses)
        best_dev_accuracy = max(validation_accuracies)
        print(f"{hidden_dim:^10} | {best_train_loss:.6f} | {best_dev_accuracy:.2%}")

    # Plot learning curves for the best-performing model
    best_hidden_dim, _, _ = max(results, key=lambda x: max(x[2]))
    best_training_losses, best_validation_accuracies = train_and_evaluate_model(best_hidden_dim, args.epochs, train_data, valid_data)

    # Plot learning curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), best_training_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss")

    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), best_validation_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")

    plt.tight_layout()
    plt.show()
