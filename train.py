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

ROOT_DIR = ""
DATA_DIR = os.path.join(ROOT_DIR, "data")
PARAMS_FILE = os.path.join(ROOT_DIR, "params","parameters.params")

VALID_RATIO = 0.1
TEST_RATIO = 0.3


################################################################################
#                                                                           DATA
################################################################################
# OPEN DATA - from all ".txt" files in the data directory
# TODO: test if appending the string from each file and then using "".join(str)
#       is more efficient.
print("LOADING DATA FROM TEXT FILES")
data_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
train = ""
test = ""
valid = ""
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
        train += s[:i_test]
        test += s[i_test:i_valid]
        valid += s[i_valid:]

print("Chars in train data: {:>15d}".format(len(train)))
print("Chars in test data:  {:>15d}".format(len(test)))
print("Chars in valid data: {:>15d}".format(len(valid)))


def random_training_batch(data, char2id, length=200, batch_size=1):
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



