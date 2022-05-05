if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


import numpy as np
from dezero import Variable
import dezero.functions as F

def main():
    x = Variable(np.array([[1,2,3], [4,5,6]]))
    y = F.sum(x, axis=0)
    y.backward()
    print(y)
    print(x.grad)

    x.cleargrad()

    x = Variable(np.random.randn(2,3,4,5))
    print(x)
    y = x.sum(keepdims=True)
    print(y.shape)


if __name__ == "__main__":
    main()