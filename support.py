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

    torch.save(model.state_dict(), filepath)
    if verbose:
        print("SAVED SNAPSHOT ({:06.3f})".format(loss))
