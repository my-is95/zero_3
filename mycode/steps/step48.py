if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
import math
import dezero
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP


def main():
    # ハイパーパラメータの設定
    max_epoch = 300 # 1エポックは用意されたデータセットを全て使用する単位
    batch_size = 30 # 1度にまとめて処理するデータ数
    hidden_size = 10
    lr = 1.0

    # データの読み込み、モデル・オプティマイザの生成
    x, t = dezero.datasets.get_spiral(train=True)
    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    data_size = len(x)
    max_iter = math.ceil(data_size / batch_size)

    for epoch in range(max_epoch):
        # データセットのインデックスのシャッフル
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            # ミニバッチの生成
            batch_index = index[i*batch_size: (i+1)*batch_size]
            batch_x = x[batch_index]
            batch_t = t[batch_index]

            # 勾配の算出　パラメータ更新
            y = model(batch_x)
            loss = F.softmax_cross_entropy_simple(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(batch_t)

        # エポックごとに学習経過を出力
        avg_loss = sum_loss / data_size
        print('epoch %d, loss %.2f' % (epoch+1, avg_loss))


if __name__ == "__main__":
    main()