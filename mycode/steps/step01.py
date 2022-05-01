import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = x ** 2
        output = Variable(y)
        return output


def main():
    data = np.array(1.0)
    x = Variable(data)
    print(x.data)

    print()

    x = Variable(np.array(10))
    f = Function()
    y = f(x)
    print(type(y))
    print(y.data)


if __name__ == "__main__":
    main()