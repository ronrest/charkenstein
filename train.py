################################################################################
#                                                            IMPORTS AND GLOBALS
################################################################################
from __future__ import print_function, division, unicode_literals
from io import open  # supports unicode encoding argument for python 2.7
import os
import unidecode

import glob

from support import random_substring_ids, str2tensor
from support import id2char, char2id, n_chars
from support import nn, torch, Variable
from model import Model

ROOT_DIR = ""
DATA_DIR = os.path.join(ROOT_DIR, "data")
PARAMS_FILE = os.path.join(ROOT_DIR, "params","parameters.params")

VALID_RATIO = 0.1
TEST_RATIO = 0.3

# MODEL HYPER-PARAMETERS
SAMPLE_LENGTH = 200
N_HIDDEN = 100
N_LAYERS = 1
EMBED_SIZE = 64
DROPOUT = 0.7
ALPHA = 0.01


################################################################################
#                                                                           DATA
################################################################################
# OPEN DATA - from all ".txt" files in the data directory
# TODO: test if appending the string from each file and then using "".join(str)
#       is more efficient.
print("LOADING DATA FROM TEXT FILES")
data_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
data_train = ""
data_test = ""
data_valid = ""
for data_file in data_files:
    with open(data_file, mode="r", encoding="utf-8") as fileObj:
        print(" -", data_file)
        s = fileObj.read()              # Read text from file
        s = unidecode.unidecode(s)  # Convert to ASCII
        
        # Indices of train valid and test sets
        file_len = len(s)
        i_valid = file_len - int(file_len*VALID_RATIO)
        i_test = i_valid  - int(file_len*TEST_RATIO)
        
        # Split the data into train valid and test sets
        data_train += s[:i_test]
        data_test += s[i_test:i_valid]
        data_valid += s[i_valid:]

print("Chars in train data: {:>15d}".format(len(data_train)))
print("Chars in test data:  {:>15d}".format(len(data_test)))
print("Chars in valid data: {:>15d}".format(len(data_valid)))


################################################################################
#                                                           SUPPORTING FUNCTIONS
################################################################################

# ==============================================================================
#                                                          RANDOM_TRAINING_BATCH
# ==============================================================================
def random_training_batch(data, char2id, length=200, batch_size=1):
    """ Returns a two torch Variables X, Y, each of size
            batch_size x (length-1)

        Where each row is a random substring from the `data` string
        represented as a sequence of character ids.
        
    """
    # Randomly sample a substring for each item in batch size
    batch = []
    for i in range(batch_size):
        batch.append(random_substring_ids(data, char2id, size=length))
    
    # Convert it to a torch Variable
    batch = Variable(torch.LongTensor(batch))

    # The input and outputs are the same, just shifted by one value
    X = batch[:, :-1]

    # Not entirely sure if it is necessary to have the underlying data
    # being a different object, but doing it to be safe.
    Y = batch[:, 1:].clone()

    return X, Y
    

# X, Y = random_training_batch(train, char2id, length=20, batch_size=5)

# ==============================================================================
#                                                                          TRAIN
# ==============================================================================
def train(model, X, Y, loss_func, optimizer):
    """ Given a model, the input and target labels representing a batch of
        sequences of characters, it performs a full training step for those
        sequences of characters, updating the model parameters based on the
        specified loss and optimizer function.
    
    Args:
        model:      (Model object)
                    The model containing the neural net architecture.
        X:          (torch Variable)
                    The input tensor of shape: [batch, sequence_length]
        Y:          (torch Variable)
                    The output labels tensor of shape: [batch, sequence_length]
        loss_func:  The torch loss function.
        optimizer:  The torch optimizer function.

    Returns:
        Returns the average loss over the batch of sequences.
    """
    # Get dimensions of input and target labels
    msg = "X and Y should be shape [n_batch, sequence_length]"
    assert len(X.size()) == len(Y.size()) == 2, msg
    batch_size, sample_length = X.size()

    # Initialize hidden state and reset the accumulated gradients
    hidden = model.init_hidden(batch_size)
    model.zero_grad()
    
    # Loop through each item in the sequence
    loss = 0
    for i in range(sample_length):
        output, hidden = model(X[:,i], hidden)
        loss += loss_func(output, Y[:,i])
    
    # Calculate gradients, and update parameter weights
    loss.backward()
    optimizer.step()
    
    # Return the average loss over the batch of sequences
    return loss.data[0] / sample_length


################################################################################
#                                                                          MODEL
################################################################################
# CREATE THE MODEL
model = Model(in_size=n_chars,
              embed_size=EMBED_SIZE,
              h_size=N_HIDDEN,
              out_size=n_chars,
              n_layers=N_LAYERS,
              dropout=DROPOUT)


# SPECIFY LOSS AND OPTIMIZER FUNCTIONS
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=ALPHA)


