"""Q-ReAlign 与 TOPIQ-IAA 小样本一致性对照（P0-1 投产验收）

用途：验证 pyiqa 的 `qrealign`（默认 Mini 0.8B）在 task='aesthetic' 下与现有
TOPIQ-IAA 的排序一致性，决定是否值得接入第二路审美分。

用法：
    python scripts/srcc_compare.py --n 8
    python scripts/srcc_compare.py --n 24 --seed 7 --device cuda --out output/srcc_compare.json

注意：
    - Q-ReAlign 必须显式传 task='aesthetic'；默认 task='quality' 是 IQA，
      与审美是不同任务，照文档不传参会拿到画质失真分。
    - CPU 上 0.8B VLM 单张约 6 分钟，GPU 上约 1 秒；样本量 GPU 建议 24-64。
    - 输出附带两张表的分位数统计，用于确定 TOPIQ_CALIBRATION 锚点
      （TOPIQ 原始 MOS 若集中在中段，需按 p5-p95 锚定到 0-10 才撑得起 0.30 权重）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _sample(n: int, seed: int) -> list[Path]:
    pool = [
        p for p in (ROOT / "uploads").rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS and "video_frames" not in p.parts
    ]
    pool.sort()
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def _ranks(values: list[float]) -> list[float]:
    """平均秩（处理并列），返回 1-based 排名"""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def _kendall(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = xs[i] - xs[j]
            b = ys[i] - ys[j]
            if a == 0 or b == 0:
                continue
            if (a > 0) == (b > 0):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="样本张数")
    ap.add_argument("--seed", type=int, default=20261104)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="output/srcc_compare.json")
    args = ap.parse_args()

    import pyiqa
    import torch
    from PIL import Image

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = _sample(args.n, args.seed)
    if not paths:
        print("uploads 下没有可用照片", file=sys.stderr)
        return 1
    print(f"样本 {len(paths)} 张（seed={args.seed}, device={args.device}）")

    print("加载 TOPIQ-IAA ...")
    t0 = time.time()
    topiq = pyiqa.create_metric("topiq_iaa", device=args.device)
    print(f"  完成 {time.time() - t0:.1f}s  score_range={topiq.score_range}")

    print("加载 Q-ReAlign（task='aesthetic'）...")
    t0 = time.time()
    qrealign = pyiqa.create_metric("qrealign", device=args.device, task="aesthetic")
    print(f"  完成 {time.time() - t0:.1f}s  score_range={qrealign.score_range}"
          f"  default_task={getattr(qrealign.net, 'default_task', '?')}")

    rows: list[dict] = []
    for idx, path in enumerate(paths, 1):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[{idx}/{len(paths)}] {path.name} 读取失败: {e}")
            continue
        t0 = time.time()
        try:
            s_topiq = max(0.0, min(10.0, float(topiq(img).item())))
        except Exception as e:
            print(f"[{idx}/{len(paths)}] TOPIQ 失败 {path.name}: {e}")
            continue
        t1 = time.time()
        try:
            s_q = float(qrealign(img).item())
        except Exception as e:
            print(f"[{idx}/{len(paths)}] Q-ReAlign 失败 {path.name}: {e}")
            continue
        t2 = time.time()
        rows.append({
            "filename": path.name,
            "path": str(path),
            "topiq": round(s_topiq, 4),
            "qrealign": round(s_q, 4),
            "qrealign_x10": round(s_q * 10, 4),
            "sec_topiq": round(t1 - t0, 2),
            "sec_qrealign": round(t2 - t1, 2),
        })
        print(f"[{idx}/{len(paths)}] {path.name:42s} TOPIQ={s_topiq:6.3f}"
              f"  Q-ReAlign={s_q:6.4f} ({s_q * 10:5.2f}/10)"
              f"  {t2 - t1:5.1f}s")

    if len(rows) < 2:
        print("有效样本不足，无法计算相关系数", file=sys.stderr)
        return 1

    xs = [r["topiq"] for r in rows]
    ys = [r["qrealign"] for r in rows]
    sp = _spearman(xs, ys)
    kd = _kendall(xs, ys)
    print()
    print(f"Spearman={sp:.4f}  Kendall_tau={kd:.4f}  n={len(rows)}")
    print(f"TOPIQ      min={min(xs):.3f} max={max(xs):.3f} span={max(xs) - min(xs):.3f}")
    print(f"Q-ReAlign  min={min(ys):.4f} max={max(ys):.4f} span={max(ys) - min(ys):.4f}")
    print(f"耗时合计   TOPIQ {sum(r['sec_topiq'] for r in rows):.1f}s"
          f"  Q-ReAlign {sum(r['sec_qrealign'] for r in rows):.1f}s")

    # 分位数（标定锚点依据）：样本 < 20 张时 p05/p95 仅作参考，不宜直接当锚点
    def _pct(vals: list[float], q: float) -> float:
        s = sorted(vals)
        if len(s) == 1:
            return s[0]
        pos = q * (len(s) - 1)
        lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)

    def _dist(vals: list[float], name: str) -> dict:
        d = {
            "min": round(min(vals), 4),
            "p05": round(_pct(vals, 0.05), 4),
            "p50": round(_pct(vals, 0.50), 4),
            "p95": round(_pct(vals, 0.95), 4),
            "max": round(max(vals), 4),
            "span": round(max(vals) - min(vals), 4),
            "stdev": round((sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / len(vals)) ** 0.5, 4),
        }
        print(f"{name:11s} " + "  ".join(f"{k}={v}" for k, v in d.items()))
        return d

    print("── 分布（锚点候选）──")
    dist = {"topiq": _dist(xs, "TOPIQ"), "qrealign": _dist(ys, "Q-ReAlign")}

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "device": args.device,
        "task": "aesthetic",
        "n": len(rows),
        "spearman": round(sp, 6),
        "kendall_tau": round(kd, 6),
        "distribution": dist,
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"明细已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
