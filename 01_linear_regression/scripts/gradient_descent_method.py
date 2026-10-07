import numpy as np
import matplotlib.pyplot as plt


def gradient_descent_method(x: np.ndarray, y: np.ndarray):
    """
    勾配法による一次関数のパラメータ推定

    Parameters
    ----------
    x : numpy.ndarray
        関数の入力値が格納された配列
    y : numpy.ndarray
        関数で推定したい値が格納された配列

    Returns
    -------
    w : numpy.float64
        一次関数の傾き（重み）
    b : numpy.float64
        一次関数の切片（バイアス）
    """
    
    # 初期パラメータ
    # ()の中に数値を書く
    w = np.float64(10)
    b = np.float64(20)

    # データ数
    n = len(x)

    # 学習率
    learning_rate = np.float64(0.01)
    # 最大繰り返し回数
    max_iteration = 10000

    # 指定された回数だけ勾配法を実行
    for i in range(max_iteration):
        # 直線回帰
        fx = w*x + b

        # 重みwの勾配計算
        dw = np.dot(x, fx - y) / n
        # バイアスbの勾配計算
        db = np.mean(fx - y)

        # パラメータ更新
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 誤差計算(平均二乗誤差)
        e = np.mean(np.pow(fx - y, 2))

        # 誤差とパラメータ更新結果を確認
        _log = 'Iteration: {:3d},  e: {:8.4f},  w: {:6.4f},  b: {:6.4f}'
        print(_log.format(i + 1, e, w, b))

    return w, b


def main():
    # サンプルデータ（x, y）
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([2, 2.8, 3.6, 4.5, 5.1, 5.9])

    # 勾配降下法で直線の係数を求める
    w, b = gradient_descent_method(x, y)
    print(f"勾配降下法で求めたパラメータ: w={w:.6f}, b={b:.6f}")

    # 勾配降下法で求めたパラメータを使ってyの値を計算(予測)
    y_fit = w * x + b

    # プロット
    plt.scatter(x, y, label="Data")
    plt.plot(x, y_fit, "r-", label=f"Fit: y={w:.2f}x+{b:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig('../outputs/gradient_descent_method.png')


if __name__ == "__main__":
    main()
