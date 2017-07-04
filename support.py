import random
import string

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




