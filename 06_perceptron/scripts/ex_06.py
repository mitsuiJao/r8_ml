import numpy as np


def load_csv(path: str):
    """CSVファイルを読み込み，ヘッダーとデータを返す関数

    numpy.genfromtxtとnumpy.loadtxtを用いてCSVを読み込む

    Parameters
    ----------
    path: str
        CSVファイルのパス

    Returns
    -------
    header : list
        ヘッダー名のリスト
    data : numpy.ndarray of float, shape (n_rows, n_cols)
        データの数値配列
    """

    # ヘッダーを取得（1行目）
    header = np.genfromtxt(path, delimiter=',', dtype=str, max_rows=1)
    header = [h.strip() for h in header.tolist()]

    # 2行目以降のデータを取得
    data = np.loadtxt(path, delimiter=',', skiprows=1)

    return header, data


class Perceptron():
    """直線の順伝播と逆伝播を計算するクラス

    self.weight: numpy.ndaaray
        ネットワークの重み
    self.bias: numpy.ndarray
        ネットワークのバイアス
    self.learning_rate: Union[float, numpy.float32]

    self.x: numpy.ndarray
        ネットワークの入力
    self.y: numpy.ndarray
        ネットワークの出力
    """
    def __init__(self, input_size, output_size, learning_rate=0.01, seed=None):
        """コンストラクタ

        Parameters
        ----------
        input_size: int
            層の入力の次元数（ユニットサイズ）
        output_size: int
            層の出力の次元数（ユニットサイズ）
        learning_rate: float
            学習率
        seed: int or None
            乱数のシード（条件を揃えて比較するときに指定）
        """

        # 乱数生成器のインスタンスを生成
        rng = np.random.default_rng(seed)
        # 重み(係数)をガウス分布に従う乱数で初期化
        self.w = rng.standard_normal((input_size, output_size)) # [TASK] 乱数を入力次元数*出力次元数のサイズで初期化
        # バイアス(切片)をガウス分布に従う乱数で初期化
        self.b = rng.standard_normal(output_size) # 乱数を出力次元数のサイズで初期化

        # 学習率を設定
        self.learning_rate = learning_rate

    def forward(self, x):
        """順伝播の関数
        Parameters
        ----------
        x: numpy.ndarray
            入力データ

        Returns
        -------
        self.y: numpy.ndarray
            出力データ
        """

        # 入力を保存しておく
        self.x = x # [TASK] 入力データxをself.xに保存

        # 順伝播y = xw + bを計算
        self.y = (self.x @ self.w) + self.b # [TASK] 順伝播の計算を実装

        return self.y

    def backward(self, grad):
        """逆伝播の関数
        Parameters
        ----------
        grad: numpy.ndarray
            逆伝播で入力された勾配情報

        Returns
        -------
        grad: numpy.ndarray
            逆伝播で入力側に与える勾配情報                
        """

        # 重みとバイアスの勾配を計算
        n = self.x.shape[0]
        g_w = (self.x.T @ grad) / n # [TASK] 重みの勾配を計算
        g_b = np.sum(grad, axis=0) / n # [TASK] バイアスの勾配を計算

        # 逆伝播する勾配を計算
        g = grad @ self.w.T # [TASK] 入力側に伝播する勾配を計算

        # 重みとバイアスの更新
        self.w = self.w - self.learning_rate * g_w # [TASK] 重みの更新式を実装
        self.b = self.b - self.learning_rate * g_b # [TASK] バイアスの更新式を実装

        return g


class MeanSquaredError():
    """平均二乗誤差関数の順伝播と逆伝播を計算するクラス
    self.diff: Union[numpy.ndarray, numpy.float32]
        入力の差分
    """
    def __init__(self):
        self.diff = None

    def forward(self, y, t):
        """順伝播
        Parameters
        ----------
        y: Union[numpy.ndarray, numpy.float32]
            ネットワークの推論値
        t: Union[numpy.ndarray, numpy.float32]
            教師データ

        Returns
        ----------
        loss: Union[numpy.ndarray, numpy.float32]
            計算した誤差の値
        """

        # 出力と教師データの差分を計算
        self.diff = y - t # [TASK] 差分を計算してself.diffに保存
        # 平均二乗誤差を計算
        loss = np.mean(self.diff ** 2) # [TASK] 平均二乗誤差の計算を実装
        return loss

    def backward(self):
        """逆順伝播
        Returns
        ----------
        grad: Union[numpy.ndarray, numpy.float32]
            逆伝播で入力側に与える勾配情報
        """

        # 勾配を計算
        g = 2 * self.diff # [TASK] 勾配の計算を実装
        return g


def train(x, t, learning_rate, max_iteration, seed=0):
    """モデル, 予測値, 誤差を返す"""
    perceptron = Perceptron(x.shape[1], 1, learning_rate, seed=seed)
    loss_function = MeanSquaredError()
    loss_list = []

    for _ in range(max_iteration):
        y = perceptron.forward(x)
        loss_list.append(loss_function.forward(y, t))
        perceptron.backward(loss_function.backward())

    return perceptron, perceptron.forward(x), loss_list


def main():
    header, data = load_csv("../data/advertising.csv")
    x = data[:, :-1]
    t = data[:, -1].reshape(-1, 1)

    # 名前 -> x, t, 予測値
    x_max, t_max = np.max(x, axis=0), np.max(t)
    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    t_mean, t_std = np.mean(t), np.std(t)
    conditions = {
        "raw": (x, t, lambda y: y),
        "normalize": (x / x_max, t / t_max, lambda y: y * t_max),
        "standardize": ((x - x_mean) / x_std, (t - t_mean) / t_std,
                        lambda y: y * t_std + t_mean),
    }

    learning_rates = [1e-7, 1e-5, 1e-3, 1e-2, 1e-1, 3e-1, 1.0]
    max_iteration = 500

    print(f"{'condition':<12}{'lr':>7}{'loss':>10}{'MSE':>10}{'iters':>7}  w, b")
    # 警告
    with np.errstate(all="ignore"):
        for name, (x_train, t_train, restore) in conditions.items():
            for lr in learning_rates:
                model, y, loss_list = train(x_train, t_train, lr, max_iteration)

                # MSE, 比較
                mse = np.mean((restore(y) - t) ** 2)
                if not mse < 1e6:
                    print(f"{name:<12}{lr:>7.0e}  diverged")
                    continue

                iters = next(i + 1 for i, v in enumerate(loss_list)
                             if v <= loss_list[-1] * 1.01)
                print(f"{name:<12}{lr:>7.0e}{loss_list[-1]:>10.4f}{mse:>10.3f}"
                      f"{iters:>7d}  {np.round(model.w.ravel(), 4)} {np.round(model.b, 4)}")
            print()


if __name__ == "__main__":
    main()
