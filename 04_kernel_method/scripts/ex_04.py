import numpy as np
import matplotlib.pyplot as plt


def generate_noisy_sine(n: int = 30, noise_std: float = 0.15, seed: int | None = 42):
    """
    ノイズ付き正弦波データセットを生成する

    Parameters
    ----------
    n : int
        サンプル数
    noise_std : float
        観測ノイズの標準偏差
    seed : int or None
        乱数シード，None の場合はシード固定なし

    Returns
    -------
    X : ndarray of shape (n, 1)
        入力特徴量
    y : ndarray of shape (n,)
        出力ターゲット
    x_true : ndarray of shape (m, 1)
        可視化用の高密度入力グリッド
    y_true : ndarray of shape (m,)
        ノイズなし正弦波の真値
    """

    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    y_clean = np.sin(X[:, 0])
    y = y_clean + rng.normal(0.0, noise_std, size=n)

    x_test = np.linspace(0.0, 2.0 * np.pi, 500).reshape(-1, 1)
    y_test = np.sin(x_test[:, 0])
    return X, y, x_test, y_test


def gaussian_kernel(x: np.ndarray, x_star: np.ndarray, beta: float) -> np.ndarray:
    """
    ガウスカーネルのグラム行列を計算する

    Parameters
    ----------
    x : ndarray of shape (n_samples_1, n_features)
        入力行列1
    x_star : ndarray of shape (n_samples_2, n_features)
        入力行列2
    length_scale : float
        長さ尺度パラメータ，値が小さいほど局所的にフィットする
        
        
        
    引数が違う beta
        
        
        

    Returns
    -------
    k : ndarray of shape (n_samples_1, n_samples_2)
        グラム行列
    """

    # 各点ごとの差の二乗和を計算するため，空の配列を計算
    #     サイズはxの行数 * x_starの行数
    k = np.empty((x.shape[0], x_star.shape[0]))
    # グラム行列を作成
    for i in range(k.shape[0]):
        for j in range(k.shape[1]):
            # 各点ごとの差の二乗和を計算(expはあとで処理)
            # k[i, j] = # [TASK]
            k[i, j] = np.sum((x[i] - x_star[j]) ** 2)

    # ガウスカーネルのグラム行列を作成
    # [TASK] k = np.exp(...)
    k = np.exp(-beta * k)

    return k


def fit_kernel_regression(k: np.ndarray, y_train: np.ndarray, lamda: float) -> np.ndarray:
    """カーネル回帰を学習し，係数ベクトルを求める

    Parameters
    ----------
    k : ndarray of shape (n_samples, n_samples)
        訓練データのグラム行列（K）
    y_train : ndarray of shape (n_samples,)
        学習データのターゲット
    lamda : float
        正則かパラメータ

    Returns
    -------
    alpha : ndarray of shape (n_samples,)
        係数ベクトル。K @ alpha = y を満たす解（ここでは簡便のため逆行列を用いる）
    """

    # 正規方程式を解く
    # alpha = # [TASK] np.linalg.inv()を使う
    # alpha = np.linalg.inv(k) @ y_train
    
    # w = np.linalg.inv(x.T @ x + lam * np.eye(degree + 1)) @ x.T @ y
    alpha = np.linalg.inv(k + lamda * np.eye(k.shape[0])) @ y_train
    return alpha


def predict_kernel_regression(k: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """学習済みモデルでの予測を行う

    Parameters
    ----------
    k : ndarray of shape (n_test, n_train)
        テストデータと訓練データ間のグラム行列（K_*）
    alpha : ndarray of shape (n_train,)
        学習で得た係数ベクトル

    Returns
    -------
    y_pred : ndarray of shape (n_test,)
        予測値
    """

    # y_pred = # [TASK] Kとalphaで計算
    y_pred = k @ alpha
    return y_pred


def plot_results(
    X_train: np.ndarray,
    y_train: np.ndarray,
    x_grid: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    学習データ，真の関数，予測曲線を可視化する

    Parameters
    ----------
    X_train : ndarray of shape (n_train, 1)
        学習入力
    y_train : ndarray of shape (n_train,)
        学習ターゲット
    x_grid : ndarray of shape (m, 1)
        可視化用の高密度入力グリッド
    y_true : ndarray of shape (m,)
        真の正弦波
    y_pred : ndarray of shape (m,)
        予測曲線

    Returns
    -------
    None
        返り値はなし
    """

    plt.figure(figsize=(6, 4))
    plt.scatter(X_train[:, 0], y_train, label="Train data", s=22)
    plt.plot(x_grid[:, 0], y_true, label="True line", linewidth=2)
    plt.plot(x_grid[:, 0], y_pred, label="Result", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim([-1.5, 1.5])
    plt.legend()
    plt.tight_layout()
    plt.savefig("../outputs/kernel_regression_result.png")


def main():
    """
    データ生成から学習と可視化までを一括実行する

    Returns
    -------
    None
        返り値はなし
    """

    # 設定
    n = 30
    noise_std = 0.15

    # 係数
    beta = 0.3
    lamda = 0.1

    # データ生成
    X_train, y_train, X_test, y_true = generate_noisy_sine(n=n, noise_std=noise_std, seed=42)

    # 学習データのグラム行列を作成
    k = gaussian_kernel(X_train, X_train, beta)
    # 学習
    alpha = fit_kernel_regression(k, y_train, lamda)

    # テストデータのグラム行列を作成
    k_star = gaussian_kernel(X_test, X_train, beta)
    # 予測
    y_pred = predict_kernel_regression(k_star, alpha)

    # 可視化
    plot_results(X_train, y_train, X_test, y_true, y_pred)


if __name__ == "__main__":
    main()
