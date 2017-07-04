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




