from get_data import *
import torch as tr

print(f"tr version: {tr.__version__}")
print(f"tr backend available: {tr.backends.mps.is_available()}")