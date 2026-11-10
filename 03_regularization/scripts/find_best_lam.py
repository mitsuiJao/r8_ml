import re
import sys
from collections import defaultdict
from pathlib import Path

LAM_RE = re.compile(r"^lam=([\d.]+):")
ROW_RE = re.compile(
    r"次数\(Degree\)=\s*(\d+):\s*Train MSE=([\d.]+),\s*Val MSE=([\d.]+)"
)


def parse(path):
    """出力ログを (lam, degree, train_mse, val_mse) のリストにパースする"""
    records = []
    lam = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = LAM_RE.match(line)
        if m:
            lam = float(m.group(1))
            continue
        m = ROW_RE.search(line)
        if m and lam is not None:
            records.append(
                (lam, int(m.group(1)), float(m.group(2)), float(m.group(3)))
            )
    return records


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "output"
    records = parse(path)
    if not records:
        sys.exit("有効なレコードが見つかりませんでした")

    best_val = min(records, key=lambda r: r[3])
    best_train = min(records, key=lambda r: r[2])

    print(f"レコード数: {len(records)}")
    print(
        f"Val MSE 最小: lam={best_val[0]:.4f}, degree={best_val[1]}, "
        f"Train MSE={best_val[2]:.4f}, Val MSE={best_val[3]:.4f}"
    )
    print(
        f"Train MSE 最小: lam={best_train[0]:.4f}, degree={best_train[1]}, "
        f"Train MSE={best_train[2]:.4f}, Val MSE={best_train[3]:.4f}"
    )

    # lam ごとに全次数の MSE を平均し，平均 Val MSE が最小の lam を求める
    by_lam = defaultdict(list)
    for lam, _, tr, va in records:
        by_lam[lam].append((tr, va))
    mean_mse = {
        lam: (sum(t for t, _ in v) / len(v), sum(x for _, x in v) / len(v))
        for lam, v in by_lam.items()
    }
    best_lam = min(mean_mse, key=lambda k: mean_mse[k][1])
    print(
        f"次数平均 Val MSE 最小: lam={best_lam:.4f}, "
        f"平均 Train MSE={mean_mse[best_lam][0]:.4f}, "
        f"平均 Val MSE={mean_mse[best_lam][1]:.4f}"
    )

    # 同値で最小となる組が複数ある場合は全て表示
    ties = [r for r in records if r[3] == best_val[3]]
    if len(ties) > 1:
        print(f"Val MSE が同値の組: {len(ties)}件")
        for lam, deg, tr, va in ties[:10]:
            print(f"  lam={lam:.4f}, degree={deg}, Val MSE={va:.4f}")


if __name__ == "__main__":
    main()
