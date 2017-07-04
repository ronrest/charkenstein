import random
# ==============================================================================
#                                                               RANDOM_SUBSTRING
# ==============================================================================
def random_substring(s, size=200):
    """ Given a string and optionally a sample size (which defaults to 200),
        it returns a random substring from s of length `size`.
    """
    i = random.randint(0, len(s) - size)
    return s[i:i + size]




