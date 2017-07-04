################################################################################
#                                                            IMPORTS AND GLOBALS
################################################################################
from __future__ import print_function, division, unicode_literals
from io import open  # supports unicode encoding argument for python 2.7
import os
import unidecode

import glob


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



