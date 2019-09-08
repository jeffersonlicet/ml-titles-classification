import os
import pandas as pd
import numpy as np
from hashlib import md5

def hash_function(w):
  return int(md5(w.encode()).hexdigest(), 16)

def hash_tokens(tokens, n):
  return [(hash_function(t) % (n - 1) + 1) for t in tokens]

