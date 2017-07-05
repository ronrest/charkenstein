import os
import random
import string
import torch
import torch.nn as nn
from torch.autograd import Variable
import time


# MAP THE VOCABULARY AND INDICES
id2char = list(string.printable)
char2id = {char: id for id, char in enumerate(id2char)}
n_chars = len(id2char)

# ==============================================================================
#                                                                    MAYBE_MKDIR
# ==============================================================================
def maybe_mkdir(path):
    """ Checks if a directory path exists on the system, if it does not, then
        it creates that directory (and any parent directories needed to
        create that directory)
    """
    if not os.path.exists(path):
        os.makedirs(path)


# ==============================================================================
#                                                           GET_PARENT_DIRECTORY
# ==============================================================================
def get_parent_directory(file):
    """ Given a file path, it returns the parent directory of that file. """
    return os.path.dirname(file)


# ==============================================================================
#                                                              MAYBE_MAKE_PARDIR
# ==============================================================================
def maybe_make_pardir(file):
    """ Takes a path to a file, and creates the necessary directory structure
        on the system to ensure that the parent directory exists (if it does
        not already exist)
    """
    maybe_mkdir(get_parent_directory(file))



# ==============================================================================
#                                                               RANDOM_SUBSTRING
# ==============================================================================
def random_substring(s, size=200):
    """ Given a string and optionally a sample size (which defaults to 200),
        it returns a random substring from s of length `size`.
    """
    i = random.randint(0, len(s) - size)
    return s[i:i + size]


# ==============================================================================
#                                                           RANDOM_SUBSTRING_IDS
# ==============================================================================
def random_substring_ids(s, char2id, size=200):
    """ Given a string s, and a dictionary that maps from character to
        an index reprensenting that character, it returns a random
        substring from the text, represented as a list of character ids.
    """
    return [char2id[char] for char in random_substring(s, size=size)]


# ==============================================================================
#                                                                     STR2TENSOR
# ==============================================================================
def str2tensor(s, char2id):
    """ Given a string, and a dictionary that maps each character to an
        integer representing the embedding index, it converts the sequence
        of characters in s to a pytorch Variable tensor of character ids.
    """
    ids = [char2id[char] for char in s]
    return (Variable(torch.LongTensor(ids)))


# ==============================================================================
#                                                                    PRETTY_TIME
# ==============================================================================
def pretty_time(t):
    """ Given an elapsed time in seconds, it returns the time as a string
        formatted as: "HH:MM:SS"
    """
    hours = int(t // 3600)
    mins = int((t % 3600) // 60)
    secs = int((t % 60) // 1)
    return "{:02d}:{:02d}:{:02d}".format(hours, mins, secs)
    

# ==============================================================================
#                                                                          TIMER
# ==============================================================================
class Timer(object):
    def __init__(self):
        """ Creates a convenient stopwatch-like timer. """
        self.start_time = 0
    
    def start(self):
        """ Start the timer """
        self.start_time = time.time()
    
    def elapsed(self):
        """ Return the number of seconds since the timer was started. """
        now = time.time()
        return (now - self.start_time)
    
    def elapsed_string(self):
        """ Return the amount of elapsed time since the timer was started as a
            formatted string in format:  "HH:MM:SS"
        """
        return pretty_time(self.elapsed())


# ==============================================================================
#                                                                  TAKE_SNAPSHOT
# ==============================================================================
def take_snapshot(model, epoch, loss, name, dir, verbose=True):
    """ Takes a snapshot of all the parameter values of a model.
    
    Args:
        model: (Model Object)
        epoch: (int)
        loss:  (float)
        name:  (str) model name
        dir:   (str) directory where snapshots will be taken
        verbose: (bool)(default=True) whether it should print out feedback.
    """
    template = "{model}_{epoch:05d}_{loss:06.3f}.params"
    filename = template.format(model=name, epoch=epoch, loss=loss)
    filepath = os.path.join(dir, filename)

    maybe_mkdir(dir)
    torch.save(model.state_dict(), filepath)
    if verbose:
        print("SAVED SNAPSHOT: {}".format(filename))


# ==============================================================================
#                                                                       GENERATE
# ==============================================================================
def generate(model, char2id, seed_str='A', length=100, exploration=0.5):
    """ Generates a new string one character at a time of specified length
        based on the models currently trained weights.

    Args:
        model:       (Model object) The pytorch model to use
        char2id:     (dict) Maps characters to character indices
        seed_str:    (str)
                     The generated text will start with this text, and
                     continue generating new characters from there onwards.
        length:      (int) Length of sequence you want to generate
        exploration: (float between (0,1] )
                     The closer the value is to 0, the more likely it is
                     to just chose the predicted char with highest probability
                     (making conservative predictions). The closer it is to 1,
                     the more likely it will explore other possibilities too
                     and increases chance of making more adventurous sequences.
    
    Returns: (str)
        The generated string of desired length.
    """
    # Store the original training mode, and turn training mode Off
    train_mode = model.training
    model.train(False)

    # Initializations
    hidden = model.init_hidden(batch_size=1)
    generated = seed_str

    # Loop through each char in seed string to prepare hidden states
    seed_input = str2tensor(seed_str, char2id)
    for char_id in range(len(seed_str) - 1):
        _, hidden = model(char_id, hidden)
    
    # CONTINUE GENERATING
    for _ in range(length):
        # Use the previous char as input to the current timestep
        last_char_id = str2tensor(generated[-1], char2id)
        output, hidden = model(last_char_id, hidden)
        
        # Redistribute the relative probability that less likely items will
        # be chosen based on exploration ratio. NOTE: this is not a
        # real probability distribution, but weights for each element.
        output_dist = output.data.view(-1).div(exploration).exp()

        # Sample from redistributed probabilities as a multinomial distribution
        chosen_id = torch.multinomial(output_dist, num_samples=1)[0]

        # Add generated character to string
        generated_char = id2char[chosen_id]
        generated += generated_char

    # Restores the original training mode
    model.train(train_mode)

    return generated