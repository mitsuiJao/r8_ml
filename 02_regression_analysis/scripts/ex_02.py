import numpy as np
import matplotlib.pyplot as plt


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


def regression_analysis(x: np.ndarray, y: np.ndarray,) -> np.ndarray:
    """回帰分析によりパラメータ推定する関数
    
    設計行列xと目的変数yに対して，正規行列によりパラメータを推定

    Parameters
    ----------
    x : numpy.ndarray of float, shape (n_samples, n_features_plus1)
        切片用の 1 列を含む設計行列
    y : numpy.ndarray of float, shape (n_samples,)
        目的変数ベクトル

    Returns
    -------
    w : numpy.ndarray of float, shape (n_features_plus1,)
        切片を含む全パラメータベクトル
    """

    w = np.linalg.inv(x.T @ x) @ x.T @ y
    return w


def predict(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """設計行列とパラメータベクトルから予測値を計算する

    Parameters
    ----------
    x : ndarray of float, shape (n_samples, n_features_plus1)
        切片列を含む設計行列
    w : ndarray of float, shape (n_features_plus1,)
        切片を含むパラメータベクトル

    Returns
    -------
    y_pred : ndarray of float, shape (n_samples,)
        予測値
    """

    y_pred = x @ w
    return y_pred


def calc_r2(y: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """予測値の評価指標R^2を計算する

    Parameters
    ----------
    y : ndarray of float, shape (n_samples,)
        実際の目的変数の値
    y_pred : ndarray of float, shape (n_samples,)
        予測値

    Returns
    -------
    r2 : float
        決定係数
    """

    ss_res = (y - y_pred).T @ (y - y_pred)
    ss_tot = (y - np.average(y)).T @ (y - np.average(y))
    r2 = 1 - ss_res / ss_tot
    return r2


def calc_rmse(y: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """予測値の評価指標 RMSE を計算する

    Parameters
    ----------
    y : ndarray of float, shape (n_samples,)
        実際の目的変数の値
    y_pred : ndarray of float, shape (n_samples,)
        予測値

    Returns
    -------
    rmse : float
        平均二乗誤差の平方根
    """

    rmse = np.sqrt((y - y_pred).T @ (y - y_pred) / len(y))
    return rmse


def main():

    # CSVファイルの読み込み
    data_path="../data/advertising.csv"
    header, data = load_csv(data_path)

    # 説明変数と目的変数に分割
    x = data[:, :-1]
    y = data[:, -1]

    # 目的変数の列名を除去(説明変数だけのリストにする)
    header = header[:-1]

    # 交互作用項（TV広告費とラジオ広告費の積）
    tv_x_radio = x[:, 0] * x[:, 1]

    # 交互作用項を説明変数に追加
    x = np.hstack([x, tv_x_radio.reshape(-1, 1)])
    header.append("TV_x_Radio")
    
    #############################################
    from itertools import combinations
    
    x_ori = x
    
    for i in range(1, 5):
        for c in list(combinations([0, 1, 2, 3], i)):
            # 説明変数を選択
            x = x_ori[:, c]

            # バイアス項：要素1の行列，xの行数×1
            bias = np.ones((x.shape[0], 1))
            # 説明変数に切片列を追加して設計行列を作成
            x = np.hstack([bias, x])

            # 回帰分析
            w = regression_analysis(x, y)
            # 切片と回帰係数に分割
            intercept = w[0]
            coefs = w[1:]

            # 予測値の計算と評価指標の算出
            y_pred = predict(x, w)
            
            
            name = [header[i] for i in c]
            print(f"{name} R2: {calc_r2(y, y_pred)}, rmse: {calc_rmse(y, y_pred)}")
    #############################################

    # 求めたパラメータの表示
    print("回帰分析結果")
    print(f"切片: {intercept}") 
    for i in range(len(coefs)):
        print(f"回帰係数({header[i]}): {coefs[i]}")
    
    print("=====================================")

    # NumPyを用いて回帰分析（確認用）
    # SVDにより安定した数値をを解を算出しているため，結果が異なる場合がある
    w, residuals, rank, s = np.linalg.lstsq(x, y, rcond=None)
    intercept = w[0]
    coefs = w[1:]
    print("NumPyによる回帰分析結果（確認用）")
    print(f"切片: {intercept}") 
    for i in range(len(coefs)):
        print(f"回帰係数({header[i]}): {coefs[i]}")

    print("=====================================")

    # 評価指標の計算と表示
    r2 = calc_r2(y, y_pred)
    rmse = calc_rmse(y, y_pred)

    print("予測値の評価指標")
    print(f"R^2: {r2}")
    print(f"RMSE: {rmse}")

    # グラフに描画(横軸が実測値，縦軸が予測値の散布図)
    # 直線上に点が分布していれば予測精度が高いことを示す
    plt.figure(figsize=(6,6))
    plt.scatter(y, y_pred, alpha=0.7)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Advertising.csv: Actual vs Predicted")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.tight_layout()
    plt.savefig('../outputs/ex_02.png')


if __name__ == "__main__":
    main()
