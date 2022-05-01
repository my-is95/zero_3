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

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        # 処理方式を再帰→ループに変更
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # リストの末尾から関数を取得（リストの末尾は削除）
            # 関数の入出力を複数に対応できるように修正
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:  # 入力の微分に値が代入されていない場合
                    x.grad = gx
                else:   # 既に入力の微分に値が代入されている場合
                    x.grad = x.grad + gx
                if x.creator is not None:
                    funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    """
    DeZeroの動的計算グラフの仕組みは、実際に計算が行われるときに、
    変数という「箱」にその「つながり」を記録することによって行われる。
    （同様のアプローチがPyTorchやChainerで行われている）
    """
    def __call__(self, *inputs):
        # 可変長の引数を扱えるように修正
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)  # アスタリスクを付けてアンパッキング
        if not isinstance(ys, tuple):   # タプルではない場合の追加対応
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        # 関数の戻り値が1つの場合は、その1つの変数を直接返すようにしている
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

class Square(Function):
    def forward(self, x):
        return x ** 2

    # gy: 出力側から伝わる微分
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx


# 自分で作成した関数クラスをPython関数のように使用するために、関数でラップする
def add(x0, x1):
    f = Add()
    return f(x0, x1)

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
    x = Variable(np.array(3.0))
    y = add(x, x)
    y.backward()
    print(x.grad)


if __name__ == "__main__":
    main()