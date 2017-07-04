import random
def random_substring(s, size=200):
    i = random.randint(0, len(s) - size)
    return s[i:i + size]




