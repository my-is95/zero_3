if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
import math
import dezero
from dezero import Variable, DataLoader, test_mode
import dezero.functions as F
from dezero.models import MLP

def main():
    x = np.ones(5)
    print(x)

    # 学習時
    y = F.dropout(x)
    print(y)

    # テスト時
    with test_mode():
        y = F.dropout(x)
        print(y)


if __name__ == "__main__":
    main()