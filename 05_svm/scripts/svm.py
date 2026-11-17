import numpy as np
import matplotlib.pyplot as plt


def generate_toy_data_nonseparable(
    n_samples_per_class=50,
    random_state=0,
    flip_y=0.005
):
    """誤判定が発生しやすい二値分類データを生成する関数

    クラスの平均を近づけて分布を重ね，一部のラベルを反転することで
    線形分離が完全には成り立たないデータを作る

    Parameters
    ----------
    n_samples_per_class : int, optional
        各クラスのサンプル数
    random_state : int, optional
        乱数シード
    flip_y : float, optional
        ラベルをランダムに反転する割合，0から1の範囲で指定

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        入力特徴量
    y : ndarray of shape (n_samples,)
        クラスラベル，値は{-1, 1}
    """
    rng = np.random.RandomState(random_state)

    # クラスの平均を近づけて，だいぶ重なりやすくする
    mean_pos = np.array([1.0, 1.0])
    mean_neg = np.array([-1.0, -1.0])

    # 共分散を大きめにして分散を広げる
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])

    X_pos = rng.multivariate_normal(mean_pos, cov, size=n_samples_per_class)
    X_neg = rng.multivariate_normal(mean_neg, cov, size=n_samples_per_class)

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n_samples_per_class), -np.ones(n_samples_per_class)])

    # ラベルをランダムに一部反転
    #     どう頑張っても完全には分離できない状況を作る
    n_samples = X.shape[0]
    n_flip = int(flip_y * n_samples)
    if n_flip > 0:
        flip_indices = rng.choice(n_samples, size=n_flip, replace=False)
        y[flip_indices] *= -1.0

    return X, y


def plot_data_and_boundary(X, y, w, b):
    """2次元データと分離境界をプロットする関数

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        入力特徴量
    y : ndarray of shape (n_samples,)
        クラスラベル，値は{-1, 1}
    w : ndarray of shape (2,)
        学習された重みベクトル
    b : float
        学習されたバイアス

    Returns
    -------
    None
    """
    pos_idx = y == 1
    neg_idx = y == -1

    plt.figure()
    plt.scatter(X[pos_idx, 0], X[pos_idx, 1], marker="o", label="class +1")
    plt.scatter(X[neg_idx, 0], X[neg_idx, 1], marker="x", label="class -1")

    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    xs = np.linspace(x_min, x_max, 200)

    if abs(w[1]) > 1e-6:
        ys = -(w[0] * xs + b) / w[1]
        plt.plot(xs, ys, label="decision boundary")
    else:
        x_line = -b / w[0]
        plt.axvline(x_line, label="decision boundary")

    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.savefig("../outputs/svm_decision_boundary.png")


def plot_loss_curve(losses):
    """エポックごとの目的関数値をプロットする関数

    Parameters
    ----------
    losses : list of float
        各エポック終了時の目的関数の値

    Returns
    -------
    None
    """
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("objective value")
    plt.tight_layout()
    plt.savefig("../outputs/svm_loss_curve.png")


def hinge_loss(w, b, X, y, lam):
    """ソフトマージンSVMの目的関数値を計算する関数

    損失は以下の形
    0.5 * ||w||^2 + lam * sum_i max(0, 1 - y_i * (w^T x_i + b))

    Parameters
    ----------
    w : ndarray of shape (n_features,)
        重みベクトル
    b : float
        バイアス項
    X : ndarray of shape (n_samples, n_features)
        入力特徴量
    y : ndarray of shape (n_samples,)
        クラスラベル，値は{-1, 1}
    lam : float
        マージン違反に対する重み

    Returns
    -------
    loss : float
        目的関数の値
    """
    # スコアを計算（wx+b）
    scores = X @ w + b
    # ヒンジ損失を計算
    hinge = np.maximum(0, 1 - y * scores)
    # 目的関数値を計算
    loss = 0.5 * np.dot(w, w) + lam * np.sum(hinge)
    return loss


def sgd_step(w, b, x_i, y_i, lr, lam):
    """一つのサンプルに対するSGD更新を行う関数

    勾配は以下を利用
    margin = y_i * (w^T x_i + b)

    margin >= 1 のとき
        grad_w = w
        grad_b = 0
    margin < 1 のとき
        grad_w = w - C * y_i * x_i
        grad_b = -C * y_i

    Parameters
    ----------
    w : ndarray of shape (n_features,)
        現在の重みベクトル
    b : float
        現在のバイアス
    x_i : ndarray of shape (n_features,)
        一つのサンプル
    y_i : float
        サンプルのラベル，値は{-1, 1}
    lr : float
        学習率
    lam : float
        マージン違反に対する重み

    Returns
    -------
    w_new : ndarray of shape (n_features,)
        更新後の重みベクトル
    b_new : float
        更新後のバイアス
    """
    # マージンを計算
    margin = y_i * (np.dot(w, x_i) + b)

    # マージンが1以上か未満かで勾配を分ける
    if margin >= 1.0:
        # マージンを満たしている場合
        # 重みとバイアスの正則化項のみが勾配になる
        grad_w = w
        grad_b = 0
    else:
        # マージンを満たしていない場合
        # 正則化項とマージン違反に対する項の両方が勾配になる
        grad_w = w - lam * y_i * x_i
        grad_b = -lam * y_i

    # パラメータ更新
    w_new = w - lr * grad_w
    b_new = b - lr * grad_b

    return w_new, b_new


def train_svm_sgd(X, y, lr=0.01, lam=1.0, n_epochs=50, random_state=0):
    """ソフトマージンSVMをSGDで学習する関数

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        入力特徴量
    y : ndarray of shape (n_samples,)
        クラスラベル，値は{-1, 1}
    lr : float, optional
        学習率
    alm : float, optional
        マージン違反に対する重み
    n_epochs : int, optional
        エポック数，データ全体を何周するか
    random_state : int, optional
        乱数シード

    Returns
    -------
    w : ndarray of shape (n_features,)
        学習された重みベクトル
    b : float
        学習されたバイアス
    losses : list of float
        各エポック終了時の目的関数の値
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape

    w = np.zeros(n_features)
    b = 0.0

    losses = []

    # 更新ループ
    for epoch in range(n_epochs):
        indices = rng.permutation(n_samples)

        # データをシャッフルして1つずつ取り出しSGDで更新
        for idx in indices:
            x_i = X[idx]
            y_i = y[idx]
            # 更新
            w, b = sgd_step(w, b, x_i, y_i, lr, lam)

        # エポック終了時の目的関数値を記録
        loss = hinge_loss(w, b, X, y, lam)
        losses.append(loss)

    return w, b, losses


def predict(X, w, b):
    """学習済みSVMで予測を行う関数

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        入力特徴量
    w : ndarray of shape (n_features,)
        学習された重みベクトル
    b : float
        学習されたバイアス

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        予測ラベル，値は{-1, 1}
    """
    # スコアを計算（wx+b）
    scores = X @ w + b
    # ラベルに変換
    #     スコアが0以上なら1，未満なら-1
    y_pred = np.where(scores >= 0.0, 1.0, -1.0)
    return y_pred


def main():
    """簡単なデータでソフトマージンSVMの学習と可視化を行うメイン関数"""
    X, y = generate_toy_data_nonseparable(n_samples_per_class=50, random_state=0)

    lr = 0.01
    lam = 0.1
    n_epochs = 50

    w, b, losses = train_svm_sgd(X, y, lr=lr, lam=lam, n_epochs=n_epochs, random_state=0)

    y_pred = predict(X, w, b)
    acc = np.mean(y_pred == y)
    print(f"Training accuracy: {acc:.3f}")

    plot_data_and_boundary(X, y, w, b)
    plot_loss_curve(losses)


if __name__ == "__main__":
    main()
