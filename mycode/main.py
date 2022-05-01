import numpy as np
import weakref
import contextlib
import unittest


class Config:
    enable_backprop = True  # 逆伝播を可能にするかどうか

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)


class Variable:
    def __init__(self, data):
        # np.ndarray 以外の型の入力を受け付けないようにバリデーションする
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None    # 逆伝播によって実際に計算された時に値を持つ
        self.creator = None
        self.generation = 0 # 逆伝播の順番を管理するために世代を管理

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()
        # 役割1: 既にfuncsリストに追加済みの関数を追加しない、役割2: 関数が追加される毎に世代順にリストを並び替え
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop() # リストの末尾から関数を取得（リストの末尾は削除）
            # 関数の入出力を複数に対応できるように修正
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:  # 入力の微分に値が代入されていない場合
                    x.grad = gx
                else:   # 既に入力の微分に値が代入されている場合
                    x.grad = x.grad + gx
                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # 参照カウントが0になり、微分のデータはメモリから消去される


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
        if Config.enable_backprop:
            # 逆伝播を正常な順番で行うために世代管理
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
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
    with no_grad():
        x = Variable(np.array(2.0))
        y = square(x)
    print(y.data)


if __name__ == "__main__":
    main()