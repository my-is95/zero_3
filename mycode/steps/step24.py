if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

def sphere(x, y):
    z = x ** 2 + y ** 2
    return z

def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

def main():
    x = Variable(np.array(1.0))
    y = Variable(np.array(1.0))
    z = sphere(x, y)
    z.backward()
    print(x.grad, y.grad)

    x1 = Variable(np.array(1.0))
    y1 = Variable(np.array(1.0))
    z1 = matyas(x1, y1)
    z1.backward()
    print(x1.grad, y1.grad)


if __name__ == "__main__":
    main()

