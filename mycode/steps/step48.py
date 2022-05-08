if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


def main():
    x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
    t = np.array([2, 0, 1, 0])
    model = MLP((10, 3))
    y = model(x)
    loss = F.softmax_cross_entropy_simple(y, t)
    print(loss)


if __name__ == "__main__":
    main()