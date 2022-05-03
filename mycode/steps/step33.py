if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable

def f(x):
    y = x**4 - 2*x**2
    return y

def main():
    print("="*5, "2回微分の例", "="*5)
    x = Variable(np.array(2.0))
    y = f(x)
    y.backward(create_graph=True)
    print(x.grad)

    gx = x.grad
    x.cleargrad()
    gx.backward()
    print(x.grad)

    print("="*5, "ニュートン法の例", "="*5)
    x = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print(i,x)
        y = f(x)
        x.cleargrad()
        y.backward(create_graph=True)
        grad1 = x.grad

        x.cleargrad()
        grad1.backward()
        grad2 = x.grad

        x.data -= grad1.data / grad2.data


if __name__ == "__main__":
    main()