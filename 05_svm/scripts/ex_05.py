import numpy as np
import matplotlib.pyplot as plt


def generate_concentric_circles(
    n_samples_per_class=100,
    radius_inner=1.0,
    radius_outer=3.0,
    noise_std=0.2,
    random_state=0,
):
    """同心円パターンの二値分類データを生成する関数

    内側と外側のリングにクラスを割り当てることで線形分離不可能なデータを作る

    Parameters
    ----------
    n_samples_per_class : int, optional
        各クラスに生成するサンプル数
    radius_inner : float, optional
        内側のリングの半径
    radius_outer : float, optional
        外側のリングの半径
    noise_std : float, optional
        半径方向に加えるガウスノイズの標準偏差
    random_state : int, optional
        乱数シード

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        入力特徴量
    y : ndarray of shape (n_samples,)
        クラスラベル，値は{-1, 1}
    """
    rng = np.random.RandomState(random_state)

    n = n_samples_per_class

    angles_inner = rng.rand(n) * 2.0 * np.pi
    angles_outer = rng.rand(n) * 2.0 * np.pi

    r_inner = radius_inner + noise_std * rng.randn(n)
    r_outer = radius_outer + noise_std * rng.randn(n)

    x_inner = np.column_stack([
        r_inner * np.cos(angles_inner),
        r_inner * np.sin(angles_inner),
    ])

    x_outer = np.column_stack([
        r_outer * np.cos(angles_outer),
        r_outer * np.sin(angles_outer),
    ])

    X = np.vstack([x_inner, x_outer])
    y = np.hstack([
        np.ones(n),
        -np.ones(n),
    ])

    return X, y


def plot_dataset(X, y, title="dataset"):
    """二次元データとクラスラベルを散布図で表示する関数

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        入力特徴量
    y : ndarray of shape (n_samples,)
        クラスラベル，値は{-1, 1}
    title : str, optional
        図のタイトル

    Returns
    -------
    None
    """
    pos = y == 1
    neg = y == -1

    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], marker="o", label="class +1")
    plt.scatter(X[neg, 0], X[neg, 1], marker="x", label="class -1")
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig("../outputs/ex_05_dataset.png")


def plot_decision_boundary_with_feature_map(X, y, w, b, feature_map, title=None):
    """特徴写像を通した線形分類器の決定境界を元の2次元空間で可視化する関数

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        元の入力特徴量
    y : ndarray of shape (n_samples,)
        クラスラベル，値は{-1, 1}
    w : ndarray of shape (n_features_mapped,)
        特徴写像後の空間で学習された重みベクトル
    b : float
        学習されたバイアス
    feature_map : callable
        X を入力として写像後特徴量を返す関数
    title : str, optional
        図のタイトル

    Returns
    -------
    None
    """
    pos = y == 1
    neg = y == -1

    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], marker="o", label="class +1")
    plt.scatter(X[neg, 0], X[neg, 1], marker="x", label="class -1")

    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    Z_feat = feature_map(grid)
    scores = Z_feat @ w + b
    scores = scores.reshape(xx.shape)

    # NaN や inf が含まれる場合は可視化をあきらめる
    if not np.all(np.isfinite(scores)):
        print("Warning: scores contain NaN or inf，skip decision region")
        plt.xlabel("x1")
        plt.ylabel("x2")
        if title is not None:
            plt.title(title)
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        return

    s_min = scores.min()
    s_max = scores.max()

    # 全て同じ値なら境界が描けないので塗りつぶしは諦める
    if s_min == s_max:
        print("Warning: scores are constant，cannot draw contour")
        plt.xlabel("x1")
        plt.ylabel("x2")
        if title is not None:
            plt.title(title)
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        return

    # 連続値として scores を塗りつぶしに使う
    levels = np.linspace(s_min, s_max, 20)
    plt.contourf(xx, yy, scores, levels=levels, alpha=0.2)

    # 決定境界 f(x)=0 を実線で描画
    plt.contour(xx, yy, scores, levels=[0.0], linestyles="--", linewidths=2)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    plt.savefig("../outputs/ex_05_decision_boundary.png")


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
    # 損失の推移を表示
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("objective value")
    plt.tight_layout()
    plt.savefig("../outputs/ex_05_loss_curve.png")


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
    y_pred = np.where(scores >= 0.0, 1.0, -1.0)
    return y_pred


def radial_quadratic_features(X):
    """2次元入力を半径方向の2次特徴に写像する関数

    各点について x1, x2, r2=x1**2+x2**2 を特徴量とする

    Parameters
    ----------
    X : ndarray of shape (n_samples, 2)
        元の入力特徴量，2次元

    Returns
    -------
    Z : ndarray of shape (n_samples, 3)
        写像後の特徴量，列は [x1, x2, x1**2 + x2**2]
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    # 半径の2乗を計算
    r2 = x1**2 + x2**2
    # 新しい特徴量行列を作成
    Z = np.column_stack([x1, x2, r2])
    return Z


def main():
    """同心円データを多項式特徴に写して線形SVMで学習するメイン関数"""
    # 元の同心円データを生成
    X, y = generate_concentric_circles(
        n_samples_per_class=100,
        radius_inner=1.0,
        radius_outer=3.0,
        noise_std=0.2,
        random_state=0,
    )

    # 特徴写像を適用
    Z = radial_quadratic_features(X)

    # SGD でソフトマージンSVMを学習
    lr = 0.01
    lam = 1
    n_epochs = 50

    w, b, losses = train_svm_sgd(Z, y, lr=lr, lam=lam, n_epochs=n_epochs, random_state=0)

    # 訓練データでの精度を計算
    y_pred = predict(Z, w, b)
    acc = np.mean(y_pred == y)
    print(f"Training accuracy: {acc:.3f}")

    # データと決定境界を可視化
    plot_dataset(X, y, title="Concentric circles dataset")
    plot_decision_boundary_with_feature_map(
        X,
        y,
        w,
        b,
        feature_map=radial_quadratic_features,
        title="SVM with radial quadratic features",
    )

    plot_loss_curve(losses)


if __name__ == "__main__":
    main()
