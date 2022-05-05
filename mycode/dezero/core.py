import numpy as np
import weakref
import contextlib

import dezero


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
    __array_priority__ = 200    # 演算子の優先度を高い値にしておくことで他のデータ型との演算時に、Variableの演算子による処理が優先される

    def __init__(self, data, name=None):
        # np.ndarray 以外の型の入力を受け付けないようにバリデーションする
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.name = name
        self.grad = None    # 逆伝播によって実際に計算された時に値を持つ
        self.creator = None
        self.generation = 0 # 逆伝播の順番を管理するために世代を管理

    # @property を付けることにより、shapeメソッドをインスタンス変数として扱うことができる
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype

    # __len__ 特殊メソッドを実装すれば、Variableインスタンスに対してlen関数を使用可能になる
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'Variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '*9)
        return 'Variable(' + p + ')'

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

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

            with using_config('enable_backprop', create_graph):
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

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self):
        return dezero.functions.transpose(self)

    @property
    def T(self):
        return dezero.functions.transpose(self)

    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)


class Function:
    """
    DeZeroの動的計算グラフの仕組みは、実際に計算が行われるときに、
    変数という「箱」にその「つながり」を記録することによって行われる。
    （同様のアプローチがPyTorchやChainerで行われている）
    """
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
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


# 足し算
class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

# 掛算
class Mul(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0 * x1, gx1 * x0

# 負数
class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

# 引き算
class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, -gx1

# 割り算
class Div(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        gx0 = gx0 / x1
        gx1 = gx1 * (-x0 / x1 ** 2)
        return gx0, gx1

# べき乗
class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        x = as_array(x)
        y = x ** self.c
        return y

    def backward(self, gy):
        x, = self.inputs
        c = self.c
        gx = c * x ** (self.c - 1) * gy
        return gx


# 自分で作成した関数クラスをPython関数のように使用するために、関数でラップする
def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)    # x1, x0 を入れ替え

def pow(x, c):
    return Pow(c)(x)


# 演算子のオーバーロードを定義するための関数
def setup_variable():
    # 演算子 のオーバーロード
    Variable.__mul__ = mul
    Variable.__rmul__ = mul # 可換のためadd関数の再利用が可能
    Variable.__add__ = add
    Variable.__radd__ = add # 可換のためadd関数の再利用が可能
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    # Variable.__div__ = div
    # Variable.__rdiv__ = rdiv
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow