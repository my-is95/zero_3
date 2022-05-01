import numpy as np
import unittest


class Variable:
    def __init__(self, data):
        # np.ndarray 以外の型の入力を受け付けないようにバリデーションする
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None    # 逆伝播によって実際に計算された時に値を持つ
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 処理方式を再帰→ループに変更
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # リストの末尾から関数を取得（リストの末尾は削除）
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator) # 1つ前の関数をリストに追加


class Function:
    """
    DeZeroの動的計算グラフの仕組みは、実際に計算が行われるときに、
    変数という「箱」にその「つながり」を記録することによって行われる。
    （同様のアプローチがPyTorchやChainerで行われている）
    """
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)    # 出力変数に生みの親を覚えさせる
        self.input = input          # 入力された変数を覚える
        self.output = output        # 出力も覚えさせる
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        return x ** 2

    # gy: 出力側から伝わる微分
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


# 自分で作成した関数クラスをPython関数のように使用するために、関数でラップする
def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)


# 中心差分近似によって、数値微分を求める
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1)) # ランダムな入力値を生成
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


def main():
    pass


if __name__ == "__main__":
    main()