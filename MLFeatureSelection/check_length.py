from MLFeatureSelection import tools
import sys
import numpy as np


def check_length(a,b):
    print(len(tools.readlog(a,b)))

if __name__ == "__main__":
    check_length(sys.argv[1],np.float(sys.argv[2]))

