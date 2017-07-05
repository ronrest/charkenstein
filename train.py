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
from support import Timer, pretty_time
from support import nn, torch, Variable
from support import generate
from model import Model

ROOT_DIR = ""
MODEL_NAME = "modelA"

DATA_DIR = os.path.join(ROOT_DIR, "data")
PARAMS_DIR = os.path.join(ROOT_DIR, "snapshots", MODEL_NAME)

VALID_RATIO = 0.1
TEST_RATIO = 0.3

# MODEL HYPER-PARAMETERS
SAMPLE_LENGTH = 200
BATCH_SIZE = 256
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
#                                                           PRINT_TRAIN_FEEDBACK
# ==============================================================================
def print_train_feedback(step, total_steps, loss, elapsed_time, avg_time):
    """ Prints a line of feedback about the training process such as:
    
         3000 (  18.8%) 00:00:02 | AVG_MS:  25.47 | LOSS:  2.421
    
          ^      ^       ^         ^                ^
          |      |       |         |                L Average Loss
          |      |       |         |
          |      |       |         L Average train time per sample
          |      |       |
          |      |       L Elapsed time
          |      |
          |      L Progress
          |
          L Step number
    """
    progress = 100 * float(step) / total_steps
    avg_time_ms = avg_time * 1000
    
    #    3000 (  18.8%) 00:00:02 | AVG_MS:  25.47 | LOSS:  2.421
    template = "{: 8d} ({: 6.1f}%) {} | AVG_MS:{: 7.2f} | LOSS:{: 7.3f}"
    print(template.format(step, progress, pretty_time(elapsed_time), avg_time_ms, loss))


# ==============================================================================
#                                                                          TRAIN
# ==============================================================================
def train(model, X, Y):
    """ Given a model, the input and target labels representing a batch of
        sequences of characters, it performs a full training step for those
        sequences of characters, updating the model parameters. based on the
        loss and optimizer of the model.
    
    Args:
        model:      (Model object)
                    The model containing the neural net architecture.
        X:          (torch Variable)
                    The input tensor of shape: [batch, sequence_length]
        Y:          (torch Variable)
                    The output labels tensor of shape: [batch, sequence_length]

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
        loss += model.loss_func(output, Y[:,i])
    
    # Calculate gradients, and update parameter weights
    loss.backward()
    model.optimizer.step()
    
    # Return the average loss over the batch of sequences
    return loss.data[0] / sample_length


# ==============================================================================
#                                                                  TRAIN_N_STEPS
# ==============================================================================
def train_n_steps(model, train_data, n_steps=1000, batch_size=32, feedback_every=1000):
    """ Trains the model for n_steps number of steps """
    total_timer = Timer()
    feedback_timer = Timer()
    total_timer.start()
    feedback_timer.start()
    
    total_loss = 0
    feedback_loss = 0
    for step in range(1, n_steps + 1):
        # Perform a training step
        X, Y = random_training_batch(train_data,char2id,SAMPLE_LENGTH,batch_size)
        loss = train(model, X, Y)
        
        # Increment losses
        total_loss += loss
        feedback_loss += loss
        
        # Print Feedback
        if (step > 0) and (step % feedback_every == 0):
            # Average Loss over feedback steps, and avg train time per sample
            avg_feedback_loss = feedback_loss / feedback_every
            avg_train_time = feedback_timer.elapsed()/feedback_every/batch_size
            
            print_train_feedback(step,
                                 total_steps=n_steps,
                                 loss=avg_feedback_loss,
                                 elapsed_time=total_timer.elapsed(),
                                 avg_time=avg_train_time)
            
            # Reset timer and loss for feedback cycle
            feedback_timer.start()
            feedback_loss = 0
    
    # Return the average loss, and average time
    avg_time = total_timer.elapsed() / n_steps / batch_size
    avg_loss = total_loss / n_steps
    return avg_loss, avg_time

# ==============================================================================
#                                                        PRINT_SAMPLE_GENERATION
# ==============================================================================
def print_sample_generation(model, char2id, seed_str="A", length=100, exploration=0.85):
    print("="*60, "\nGENERATED SAMPLE\n", "="*60, sep="")
    print(generate(model, char2id, seed_str="A", length=100, exploration=0.85))
    print("="*60)



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

model.update_learning_rate(ALPHA)



################################################################################
#                                                                TRAIN THE MODEL
################################################################################
num_epochs = 10
# Technically the following calculation for `samples_per_epoch` is incorrect,
# since we are randomly sampling, and not doing a sliding window of the samples
# but it is still a useful approximation to use.
samples_per_epoch = int(len(data_train))
steps_per_epoch = int(samples_per_epoch / BATCH_SIZE)
feedbacks_per_epoch = 10

try:
    for i in range(num_epochs):
        print("="*60)
        print("EPOCH ", i)
        print("="*60)
        train_n_steps(model,
                      data_train,
                      n_steps=steps_per_epoch,
                      batch_size=BATCH_SIZE,
                      feedback_every=int(steps_per_epoch / feedbacks_per_epoch))
        
        # TODO: evaluate on validation data
        # TODO: take snapshots
        # TODO: printouts
        # Print a sample of generated text
        print_sample_generation(model, char2id, exploration=0.85)

        # TODO: overall progress
        # TODO: elapsed time

except KeyboardInterrupt:
    print("\n A keyboard interupt wsa deected. ")
    print("TODO: Save the model")

print("DONE!!!")
