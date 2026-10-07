import numpy as np
import matplotlib.pyplot as plt


def gradient_descent_quadratic(x: np.ndarray, y: np.ndarray):
    """
    勾配法による二次関数のパラメータ推定

    Parameters
    ----------
    x : numpy.ndarray
        関数の入力値が格納された配列
    y : numpy.ndarray
        関数で推定したい値が格納された配列

    Returns
    -------
    a : numpy.float64
        二次関数の係数（x^2の項）
    b : numpy.float64
        二次関数の係数（xの項）
    c : numpy.float64
        二次関数の切片
    """
    # 初期パラメータ
    a = np.float64(1)
    b = np.float64(5)
    c = np.float64(40)

    # データ数
    n = len(x)

    # 学習率
    learning_rate = np.float64(0.001)
    # 最大繰り返し回数
    max_iteration = 10000

    for i in range(max_iteration):
        # 予測値
        fx = a * np.pow(x, 2) + b * x + c
                
        # 勾配計算
        da = np.dot(np.pow(x, 2), fx - y) / n
        db = np.dot(x, fx - y) / n
        dc = np.mean(fx - y)

        # パラメータ更新
        a = a - learning_rate * da
        b = b - learning_rate * db
        c = c - learning_rate * dc

        # 誤差計算(平均二乗誤差)
        e = np.mean(np.pow(fx - y, 2))

        # ログ出力
        if (i + 1) % 100 == 0 or i == 0:
            _log = 'Iteration: {:4d},  e: {:10.6f},  a: {:8.4f},  b: {:8.4f},  c: {:8.4f}'
            print(_log.format(i + 1, e, a, b, c))

    return a, b, c


def main():
    # サンプルデータ（x, y）: ノイズなしの二次関数
    x = np.array([0, 1, 2, 3, 4, 5])
    y = 1.2 * x**2 - 0.8 * x + 2.5 + np.array([0.5, -0.2, 0.1, -0.3, 0.4, -0.1])

    # 勾配法で二次関数の係数を求める
    a, b, c = gradient_descent_quadratic(x, y)
    print(f"勾配法で求めたパラメータ: a={a:.6f}, b={b:.6f}, c={c:.6f}")

    # 予測値
    y_fit = a * x**2 + b * x + c

    # プロット
    plt.scatter(x, y, label="Data")
    plt.plot(x, y_fit, "r-", label=f"Fit: y={a:.2f}x²+{b:.2f}x+{c:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig('../outputs/ex_01.png')


if __name__ == "__main__":
    main()
