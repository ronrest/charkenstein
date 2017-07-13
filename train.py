################################################################################
#                                                            IMPORTS AND GLOBALS
################################################################################
from __future__ import print_function, division, unicode_literals
from io import open  # supports unicode encoding argument for python 2.7
import os
import unidecode

import matplotlib.pyplot as plt
import numpy as np
import glob

from support import random_substring_ids, str2tensor
from support import pickle2obj, obj2pickle, dict2file
from support import save_hyper_params, load_hyper_params
from support import take_snapshot, epoch_snapshot, load_latest_snapshot
from support import id2char, char2id, n_chars
from support import Timer, pretty_time
from support import nn, torch, Variable
from support import generate
from model import Model
from eval import eval_model


ROOT_DIR = ""
MODEL_NAME = "modelA"

DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models", MODEL_NAME)
SNAPSHOTS_DIR = os.path.join(MODELS_DIR, "snapshots")
EVALS_FILE = os.path.join(MODELS_DIR, "evals.pickle")
HYPERPARAMS_FILE = os.path.join(MODELS_DIR, "hyperparams.txt")
LEARNING_CURVES_FILE = os.path.join(MODELS_DIR, "learning_curves.png")

VALID_RATIO = 0.1
TEST_RATIO = 0.3


################################################################################
#                                                           SUPPORTING FUNCTIONS
################################################################################

# ==============================================================================
#                                                                      LOAD_DATA
# ==============================================================================
def load_data(data_dir, test_ratio=0.3, valid_ratio=0.1):
    """ Given a directory, it opens the data from all ".txt" files
        in that directory.

    Args:
        data_dir:   (str) path to the directory containing txt files.
        test_ratio: (float)(default=0.3)
                    Portion of each file to be assigned to the test set.
        valid_ratio: (float)(default=0.1)
                    Portion of each file to be assigned to the validation set.
    Returns: (tuple of 3 strings)
        data_train, data_test, data_valid
    """
    # TODO: test if appending the string from each file to a list and
    #       then using "".join(str) is more efficient.
    # TODO: Use a better way of splitting the train, test, valid sets.
    #       At the moment it creates quite arbitrary cutoff points between
    #       samples from subsequent texts, which could be mid sentence and
    #       mid word. This is particularly an issue for validation and test sets
    #       which are relatively smaller, so will be evaluated on a lot of text
    #       segments that does not flow properly.
    print("LOADING DATA FROM TEXT FILES")
    data_files = glob.glob(os.path.join(data_dir, "*.txt"))
    data_train = ""
    data_test = ""
    data_valid = ""
    for data_file in data_files:
        with open(data_file, mode="r", encoding="utf-8") as fileObj:
            print(" -", data_file)
            s = fileObj.read()  # Read text from file
            s = unidecode.unidecode(s)  # Convert to ASCII
            
            # Indices of test and valid sets
            file_len = len(s)
            i_valid = file_len - int(file_len * valid_ratio)
            i_test = i_valid - int(file_len * test_ratio)
            
            # Split the data into train valid and test sets
            data_train += s[:i_test]
            data_test += s[i_test:i_valid]
            data_valid += s[i_valid:]
    
    print("Chars in train data: {:>15d}".format(len(data_train)))
    print("Chars in test data:  {:>15d}".format(len(data_test)))
    print("Chars in valid data: {:>15d}".format(len(data_valid)))
    
    return data_train, data_test, data_valid


# ==============================================================================
#                                                           PLOT_LEARNING_CURVES
# ==============================================================================
def plot_learning_curves(evals, file, model_name=""):
    """ Given an evals dictionary it plots the training curves and saves them
        to the ddesired file.
    """
    green = "#73AD21"
    blue = "#307EC7"
    orange = "#E65C00"
    
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(11, 6))
    fig.suptitle('Learning curves ' + model_name, y=1.000, fontsize=15)
    x_labels = map(pretty_time, np.cumsum(evals["train_time"]))
    
    ax1.set_title("Loss")
    ax1.plot(evals["train_loss"], color=orange, label="train")
    ax1.plot(evals["valid_loss"], color=blue, label="valid")
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_xticklabels(x_labels,
                        rotation=-45,  # Rotation
                        ha="left",  # Text alignment
                        fontsize=10
                        )
    
    ax2.set_title("Alpha")
    ax2.plot(evals["alpha"], color=green)
    ax2.set_yscale('log')
    ax2.set_xticklabels(x_labels,
                        rotation=-45,  # Rotation
                        ha="left",  # Text alignment
                        fontsize=10
                        )
    
    fig.savefig(file)


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
def train_n_steps(model, hyper, train_data, n_steps=1000, batch_size=32, feedback_every=1000):
    """ Trains the model for n_steps number of steps.
        Returns a tuple:
            avg_loss, total_train_time
    """
    total_timer = Timer()
    feedback_timer = Timer()
    total_timer.start()
    feedback_timer.start()
    
    total_loss = 0
    feedback_loss = 0
    for step in range(1, n_steps + 1):
        # Perform a training step
        X, Y = random_training_batch(train_data,char2id,hyper["SAMPLE_LENGTH"],batch_size)
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
    
    # Return the average loss, and total time
    avg_loss = total_loss / n_steps
    return avg_loss, total_timer.elapsed()


# ==============================================================================
#                                                        PRINT_SAMPLE_GENERATION
# ==============================================================================
def print_sample_generation(model, char2id, seed_str="A", length=100, exploration=0.85):
    print("-"*60, "\nGENERATED SAMPLE\n", "-"*60, sep="")
    print(generate(model, char2id, seed_str=seed_str, length=100, exploration=0.85))
    print("."*60)


# ==============================================================================
#                                                                 TRAIN_N_EPOCHS
# ==============================================================================
def train_n_epochs(model, hyper, data, data_valid, evals, n_epochs,
                   feedbacks_per_epoch=10, alpha_decay=1.0):
    """ Train the model for a desired amount of epochs.
        Automatically takes snapshots of the parameters after each epoch, and
        monitors the progress.

    Args:
        model:
        data:       (str) Training data
        hyper:      (dict) hyperparameters dictionary
        data_valid: (str) Validation data
        evals:      (dict of lists)
                    The dict that stores the losses and times for each epoch
        n_epochs:   (int) Number of epochs to train for
        feedbacks_per_epoch: (int) Max number of progress printouts per epoch
        alpha_decay: (float)(default=1.0)
                    How much to decay the alpha by after each epoch.

    Returns: (dict)
        - evals - the dictionary that monitors the losses, and times
    """
    timer = Timer()
    timer.start()
    
    # CALCULATE NUMBER OF STEPS NEEDED
    # Technically the following calculation for `samples_per_epoch` is incorrect,
    # since we are randomly sampling windows, and not dividing the data into an
    # even number of chunks.
    # But it is still a useful approximation, that allows us to have more variation
    # in the training data.
    samples_per_epoch = int(len(data_train) // hyper["SAMPLE_LENGTH"])
    steps_per_epoch = int(samples_per_epoch / hyper["BATCH_SIZE"])
    feedback_every = int(steps_per_epoch / feedbacks_per_epoch)
    
    try:
        for i in range(n_epochs):
            print()
            print("=" * 60)
            print("EPOCH {}/{} ({:0.2f}%) alpha={}".format(i+1, n_epochs,
                                                100 * (i / n_epochs),
                                                model.alpha))
            print("=" * 60)
            
            # TRAIN OVER A SINGLE EPOCH
            train_loss, epoch_time = train_n_steps(model,
                                                   hyper,
                                                   data_train,
                                                   n_steps=steps_per_epoch,
                                                   batch_size=hyper["BATCH_SIZE"],
                                                   feedback_every=feedback_every)
            
            evals["train_loss"].append(train_loss)
            evals["train_time"].append(epoch_time)
            evals["alpha"].append(model.alpha)
            
            # EVALUATE ON VALIDATION DATA
            eval_loss, eval_time = eval_model(model, data_valid, char2id,
                                              seq_length=hyper["SAMPLE_LENGTH"],
                                              batch_size=hyper["BATCH_SIZE"])
            evals["valid_loss"].append(eval_loss)
            evals["valid_time"].append(eval_time)
            
            # PREPARE MODEL FOR NEXT EPOCH
            model.update_learning_rate(model.alpha * alpha_decay)
            hyper["LAST_ALPHA"] = model.alpha

            # TAKE SNAPSHOTS - of parameters and evaluation dictionary
            global_epoch = len(evals["train_loss"])
            epoch_snapshot(model, epoch=global_epoch, loss=eval_loss, name=MODEL_NAME,
                           dir=SNAPSHOTS_DIR)
            obj2pickle(evals, EVALS_FILE)
            save_hyper_params(hyper, HYPERPARAMS_FILE)
            
            # FEEDBACK PRINTOUTS
            # TODO: Save a sample numerous generated strings to files at each epoch
            # Print a sample of generated text
            print_sample_generation(model, char2id, exploration=0.85)
            epoch_template = "({}) TRAIN_LOSS={: 7.3f} VALID_LOSS={: 7.3f}"
            print(epoch_template.format(timer.elapsed_string(), train_loss,
                                        eval_loss))
            
            # UPDATE LEARNING CURVE PLOT
            plot_learning_curves(evals, file=LEARNING_CURVES_FILE,
                                 model_name=MODEL_NAME)


        print("- DONE")
        return evals
    
    # HANDLE EARLY TERMINATION
    except KeyboardInterrupt:
        print("\n A keyboard interrupt was triggered at",
              timer.elapsed_string())
        
        # Save parameters as a recovery file
        print("Storing Recovery parameters")
        file = os.path.join(SNAPSHOTS_DIR, MODEL_NAME + ".recovery_params")
        take_snapshot(model, file)
        
        # Save evals as a recovery file
        print("Storing Recovery evals")
        file = os.path.join(MODELS_DIR, MODEL_NAME + ".recovery_evals")
        obj2pickle(evals, file)
        
        # Save hyper parameters
        print("Saving Hyper Params")
        hyper["LAST_ALPHA"] = model.alpha
        save_hyper_params(hyper, HYPERPARAMS_FILE)
        
        print("OK DONE")


################################################################################
#                                                                TRAIN THE MODEL
################################################################################
# LOAD DATA
data_train, data_test, data_valid = load_data(DATA_DIR, TEST_RATIO, VALID_RATIO)

# MODEL HYPER-PARAMETERS
hyper = load_hyper_params(HYPERPARAMS_FILE)

# CREATE THE MODEL
model = Model(in_size=n_chars,
              embed_size=hyper["EMBED_SIZE"],
              h_size=hyper["N_HIDDEN"],
              out_size=n_chars,
              n_layers=hyper["N_LAYERS"],
              dropout=hyper["DROPOUT"])

model.update_learning_rate(hyper["LAST_ALPHA"])

# LOAD PREVIOUSLY SAVED MODEL PARAMETERS - if they exist
load_latest_snapshot(model, SNAPSHOTS_DIR)

# KEEP TRACK OF EVALS - loading from file if they already exist
if os.path.exists(EVALS_FILE):
    print("LOADING EXISTING EVALS")
    evals = pickle2obj(EVALS_FILE)
else:
    print("INITIALIZING NEW EVALS")
    evals = {"train_loss": [],
             "valid_loss": [],
             "train_time": [],
             "valid_time": [],
             "alpha": [],
             }

print("#"*60)
print("Training {}".format(MODEL_NAME))
print("#"*60)
evals = train_n_epochs(model, hyper, data_train, data_valid, evals, n_epochs=4, alpha_decay=0.90)

print("total train time for {} epochs: {}".format(len(evals["train_time"]), pretty_time(sum(evals["train_time"]))))
print("Latest losses T: {} V: {}".format(evals["train_loss"][-1], evals["valid_loss"][-1]))


