if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from dezero import Variable
import dezero.functions as F


def main():
    x0 = Variable(np.array([1, 2, 3]))
    x1 = Variable(np.array([10]))
    y = x0 + x1
    print(y)

    y.backward()
    print(x0.grad)
    print(x1.grad)


if __name__ == "__main__":
    main()