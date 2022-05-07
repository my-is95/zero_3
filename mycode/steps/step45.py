if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L


def main():
    # データセット
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    lr = 0.2
    max_iter = 10000
    hidden_size = 10

    # モデルの定義
    class TwoLayerNet(Model):
        def __init__(self, hidden_size, out_size):
            super().__init__()
            self.l1 = L.Linear(hidden_size)
            self.l2 = L.Linear(out_size)

        def forward(self, x):
            y = self.l1(x)
            y = F.sigmoid(y)
            y = self.l2(y)
            return y

    model = TwoLayerNet(hidden_size, 1)

    # 学習の開始
    for i in range(max_iter):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data  # Layerが入れ子構造でも再帰的にパラメータを探索

        if i % 1000 == 0:
            print(loss)    

    plt.scatter(x, y, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    t = np.arange(0, 1, .01)[:, np.newaxis]
    y_pred = model(t)
    plt.plot(t, y_pred.data, color='r')
    plt.show()


if __name__ == "__main__":
    main()