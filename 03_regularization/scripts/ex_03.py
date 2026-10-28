import numpy as np
import matplotlib.pyplot as plt


def make_data(n=80, noise_std=0.25, seed=0):
    """真の関数 y = sin(2πx) にノイズを加えたデータを生成

    Parameters
    ----------
    n : int
        サンプル数，デフォルトは80
    noise_std : float
        観測ノイズの標準偏差，デフォルトは0.25
    seed : int
        乱数シード，デフォルトは0

    Returns
    -------
    X : ndarray, shape (n,)
        入力 x のサンプル，区間 [0, 1] から一様にサンプリング
    y : ndarray, shape (n,)
        ノイズを含む観測値
    y_true : ndarray, shape (n,)
        ノイズなしの真の関数値
    """

    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=n)
    y_true = np.sin(2 * np.pi * X)
    y = y_true + rng.normal(0.0, noise_std, size=n)
    return X, y, y_true


def train_val_split(X, y, val_ratio=0.3, seed=42):
    """訓練データと検証データに分割する，シャッフルしてから先頭を検証に割当

    Parameters
    ----------
    X : array_like, shape (n,)
        入力データ
    y : array_like, shape (n,)
        出力データ
    val_ratio : float
        検証データに割り当てる比率，デフォルトは0.3
    seed : int
        乱数シード，デフォルトは42

    Returns
    -------
    X_train, y_train, X_val, y_val : ndarray
        訓練用と検証用に分割されたデータ
    """

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n_val = int(len(X) * val_ratio)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def make_vander(x, degree):
    """次数 degree の多項式用 Vandermonde 行列を生成する，列は 1, x, x^2, ..., x^degree の順

    Parameters
    ----------
    x : array_like, shape (n,)
        入力値
    degree : int
        多項式の次数

    Returns
    -------
    v : ndarray, shape (n, degree+1)
        Vandermonde 行列
    """

    v = # [TASK] バンドルモンド行列を作成，np.vanderを使用
    return v


def fit_poly_lstsq(x, y):
    """多項式係数を最小二乗で推定する，NumPy の lstsq を利用する

    Parameters
    ----------
    x : ndarray, shape (n, degree+1)
        デザイン行列（Vandermonde 行列）
    y : array_like, shape (n,)
        出力データ

    Returns
    -------
    w : ndarray, shape (degree+1,)
        推定された係数ベクトル
    """

    w, *_ = # [TASK] 最小二乗法で多項式係数を推定，np.linalg.lstsqを使用
    return w


def fit_poly_ridge(x, y, lam=0.0):
    """Ridge 回帰（L2 正則化）で多項式係数を推定する，正規方程式の閉形式解を用いる

    Parameters
    ----------
    x : ndarray, shape (n, degree+1)
        デザイン行列，Vandermonde 行列を想定する
    y : array_like, shape (n,)
        出力データ
    lam : float
        正則化パラメータ λ，デフォルトは0.0
        0.0 のときは最小二乗法と同等

    Returns
    -------
    w : ndarray, shape (degree+1,)
        推定された係数ベクトル
    """

    I = # [TASK] 単位行列を作成，np.eyeを使用
    w = # [TASK] Ridge回帰で重みを計算
    return w


def predict_poly(x, w):
    """係数 w を用いて多項式の予測値を計算する

    Parameters
    ----------
    x : array_like
        予測対象の入力値，スカラーまたは配列を与えられる
    w : array_like, shape (degree+1,)
        多項式の係数ベクトル

    Returns
    -------
    y_pred : ndarray
        予測値
    """
    y_pred = # [TASK] 予測値を計算，xとwの内積
    return y_pred


def mse(y_true, y_pred):
    """平均二乗誤差を計算する

    Parameters
    ----------
    y_true : array_like
        真の値
    y_pred : array_like
        予測値

    Returns
    -------
    float
        平均二乗誤差
    """
    return # [TASK] 平均二乗誤差を計算して返す


def main():
    X, y, _ = make_data(n=80, noise_std=0.25, seed=0)
    _X_train, y_train, _X_val, y_val = train_val_split(X, y, val_ratio=0.3, seed=42)

    # L2正則化の強さ
    lam = # [TASK] 正則化の強さを決める

    # 次数を1から20まで変化させて比較するループ
    for degree in range(1, 21):
        # バンドルモンド行列を作成
        X_train = make_vander(_X_train, degree)
        X_val = make_vander(_X_val, degree)

        # 正則化なし（最小二乗法）
        w_ls = fit_poly_lstsq(X_train, y_train)
        # L2正則化（Ridge）
        w_ridge = fit_poly_ridge(X_train, y_train, lam=lam)

        # MSEを計算
        train_mse_ls = mse(y_train, predict_poly(X_train, w_ls))
        val_mse_ls = mse(y_val, predict_poly(X_val, w_ls))

        train_mse_ridge = mse(y_train, predict_poly(X_train, w_ridge))
        val_mse_ridge = mse(y_val, predict_poly(X_val, w_ridge))

        print(
            f"Degree={degree:2} | LS: Train MSE={train_mse_ls:.4f}, Val MSE={val_mse_ls:.4f} "
            f"| Ridge(λ={lam}): Train MSE={train_mse_ridge:.4f}, Val MSE={val_mse_ridge:.4f}"
        )

        # フィット可視化（両者を同一図で表示）
        _xs = np.linspace(0.0, 1.0, 400)
        xs = make_vander(_xs, degree)
        true_y = np.sin(2 * np.pi * _xs)

        plt.figure(figsize=(6, 4))
        plt.scatter(_X_train, y_train, alpha=0.7, label='Train')
        plt.scatter(_X_val, y_val, alpha=0.7, label='Val')
        plt.plot(_xs, true_y, linewidth=2, label='True f(x) = sin(2πx)')
        plt.plot(_xs, predict_poly(xs, w_ls),  linewidth=1.5, label='LS Predicted')
        plt.plot(_xs, predict_poly(xs, w_ridge), linewidth=1.5, linestyle='--', label=f'Ridge(λ={lam})')
        plt.ylim([-1.5, 1.5])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'../outputs/ridge_poly_fits_deg_{degree}.png')
        plt.clf()
        plt.close('all')


if __name__ == "__main__":
    main()
