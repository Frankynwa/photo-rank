"""TOPIQ 通道标定：全量实测原始 MOS 分布 + 反推 L1 客观分 + 重算融合分分布

用途：为两个配置决策提供数据依据
  1. `config.TOPIQ_CALIBRATION` 锚点 — 原始 MOS 集中在 4.2~5.7，不标定则 0.30 权重
     对 final_score 的方差贡献极小（实测 24 张 span 仅 1.53）；
  2. 三桶分级阈值 — 必须基于改配置后的真实分布，不能沿用批次归一化时代的分布。

做法：
  - TOPIQ 只看图片本身，与 VLM 无关 → 可独立全量重算，不消耗任何 API；
  - L1 客观分（objective）无法从 full_ranking 直接读到，用旧融合公式反推：
        final = w_vlm·q_vlm + w_topiq·topiq10 + w_obj·obj10
    其中 w_* 为 VLM 已评分时的 _effective_weights 结果（face 归零重分配）；
  - VLM 分沿用旧值（批次拉伸后的），不做改写，只替换 TOPIQ 通道观察分布迁移。

用法：
    python scripts/calibrate_topiq.py                    # 全量
    python scripts/calibrate_topiq.py --limit 100        # 抽样快速验证
    python scripts/calibrate_topiq.py --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 旧融合权重（VLM 已评分 → face 权重归零后按比例重分配，见 pipeline._effective_weights）
_W_VLM = 0.45 + 0.15 * (0.45 / 0.85)
_W_TOPIQ = 0.30 + 0.15 * (0.30 / 0.85)
_W_OBJ = 0.10 + 0.15 * (0.10 / 0.85)

# 平台默认维度权重（旧 full_ranking 生成时未启用题材化覆盖）
_DIM_W = {"composition": 0.35, "color": 0.30, "lighting": 0.20}
_DIM_W_SUM = sum(_DIM_W.values())


def _pct(vals: list[float], q: float) -> float:
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _summary(vals: list[float]) -> dict:
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {
        "n": n,
        "min": round(min(vals), 4),
        "p01": round(_pct(vals, 0.01), 4),
        "p05": round(_pct(vals, 0.05), 4),
        "p10": round(_pct(vals, 0.10), 4),
        "p25": round(_pct(vals, 0.25), 4),
        "p50": round(_pct(vals, 0.50), 4),
        "p75": round(_pct(vals, 0.75), 4),
        "p90": round(_pct(vals, 0.90), 4),
        "p95": round(_pct(vals, 0.95), 4),
        "p99": round(_pct(vals, 0.99), 4),
        "max": round(max(vals), 4),
        "span": round(max(vals) - min(vals), 4),
        "mean": round(mean, 4),
        "stdev": round(var ** 0.5, 4),
    }


def _histo(vals: list[float], lo: float, hi: float, bins: int = 12) -> list[dict]:
    """等宽直方图，看分布形状（是否双峰/长尾）"""
    step = (hi - lo) / bins if hi > lo else 1.0
    counts = [0] * bins
    for v in vals:
        i = int((v - lo) / step) if step else 0
        counts[min(max(i, 0), bins - 1)] += 1
    return [
        {"range": f"{lo + i * step:.2f}~{lo + (i + 1) * step:.2f}", "count": c}
        for i, c in enumerate(counts)
    ]


def _bucket_counts(vals: list[float], thresholds: list[float]) -> dict:
    out = {}
    for t in thresholds:
        out[f">={t}"] = sum(1 for v in vals if v >= t)
    for t in thresholds:
        out[f"<{t}"] = sum(1 for v in vals if v < t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default="output/xiaohongshu/full_ranking.json")
    ap.add_argument("--out", default="output/topiq_calibration.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="0 = 全量")
    ap.add_argument("--anchor-lo-q", type=float, default=0.02, help="标定下锚分位")
    ap.add_argument("--anchor-hi-q", type=float, default=0.98, help="标定上锚分位")
    args = ap.parse_args()

    import pyiqa
    import torch
    from PIL import Image

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ranking_path = ROOT / args.ranking
    rows = json.loads(ranking_path.read_text(encoding="utf-8"))
    if args.limit:
        rows = rows[: args.limit]
    print(f"记录 {len(rows)} 条（device={args.device}）")

    print("加载 TOPIQ-IAA ...")
    t0 = time.time()
    metric = pyiqa.create_metric("topiq_iaa", device=args.device)
    print(f"  完成 {time.time() - t0:.1f}s  score_range={metric.score_range}")

    from core.image_cache import get_cached_path

    mos: list[float] = []
    obj10: list[float] = []
    q_vlm: list[float] = []
    topiq_old: list[float] = []
    final_old: list[float] = []
    failed: list[str] = []

    t_all = time.time()
    for idx, r in enumerate(rows, 1):
        try:
            img = Image.open(get_cached_path(r["path"])).convert("RGB")
            s = max(0.0, min(10.0, float(metric(img).item())))
        except Exception as e:
            failed.append(f"{r.get('filename')}: {e}")
            continue

        dims = (
            r["composition_score"] * _DIM_W["composition"]
            + r["color_score"] * _DIM_W["color"]
            + r["lighting_score"] * _DIM_W["lighting"]
        ) / _DIM_W_SUM
        q = 0.7 * dims + 0.3 * r["vlm_overall_score"]
        o = (r["final_score"] - _W_VLM * q - _W_TOPIQ * r["overall_score"]) / _W_OBJ

        mos.append(s)
        q_vlm.append(q)
        topiq_old.append(r["overall_score"])
        final_old.append(r["final_score"])
        obj10.append(o)

        if idx % 50 == 0:
            el = time.time() - t_all
            print(f"  {idx}/{len(rows)}  已用 {el:.0f}s  预计剩余 {el / idx * (len(rows) - idx):.0f}s")

    if not mos:
        print("全部失败，无法标定", file=sys.stderr)
        return 1

    print(f"\nTOPIQ 推理完成 {len(mos)} 张，耗时 {time.time() - t_all:.1f}s；失败 {len(failed)}")

    # ── 分布统计 ──
    mos_sum = _summary(mos)
    old_sum = _summary(topiq_old)
    print("\n── TOPIQ 原始 MOS ──")
    print(json.dumps(mos_sum, ensure_ascii=False))
    print("── 旧 overall_score（p5-p95 归一化后）──")
    print(json.dumps(old_sum, ensure_ascii=False))
    print("原始 MOS 直方图:", json.dumps(_histo(mos, mos_sum["min"], mos_sum["max"]), ensure_ascii=False))

    # 反推 L1 客观分健全性检查：越界说明公式或字段假设有误
    oob = sum(1 for v in obj10 if v < -0.01 or v > 10.01)
    print(f"\n反推 L1 客观分越界 {oob}/{len(obj10)}（应为 0，否则公式假设有误）")
    print("L1 客观分分布:", json.dumps(_summary(obj10), ensure_ascii=False))

    # ── 候选锚点方案 ──
    lo_q, hi_q = args.anchor_lo_q, args.anchor_hi_q
    anchors = {
        "A_不标定": None,
        "B_分位锚": (round(_pct(mos, lo_q), 3), round(_pct(mos, hi_q), 3)),
        "C_极值锚": (round(mos_sum["min"], 3), round(mos_sum["max"], 3)),
        "D_语义锚_4_6": (4.0, 6.0),
    }

    def _apply_anchor(s: float, anchor) -> float:
        if anchor is None:
            return s
        lo, hi = anchor
        if hi <= lo:
            return s
        return max(0.0, min(1.0, (s - lo) / (hi - lo))) * 10.0

    # ── 重算融合分（只换 TOPIQ 通道，VLM/L1 不动）──
    schemes: dict[str, dict] = {}
    thresholds = [8.5, 8.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5, 4.0]
    for name, anchor in anchors.items():
        finals = [
            _W_VLM * q + _W_TOPIQ * _apply_anchor(s, anchor) + _W_OBJ * o
            for q, s, o in zip(q_vlm, mos, obj10)
        ]
        schemes[name] = {
            "anchor": list(anchor) if anchor else None,
            "final": _summary(finals),
            "buckets": _bucket_counts(finals, thresholds),
            "histo": _histo(finals, min(finals), max(finals)),
        }
        print(f"\n── 融合分分布 · {name}  anchor={anchor} ──")
        print(json.dumps(schemes[name]["final"], ensure_ascii=False))
        print("分桶:", json.dumps(schemes[name]["buckets"], ensure_ascii=False))

    # 旧配置对照（验证反推链路一致）
    base = _summary(final_old)
    print("\n── 融合分分布 · 旧配置对照（应等于现有 final_score）──")
    print(json.dumps(base, ensure_ascii=False))
    print("分桶:", json.dumps(_bucket_counts(final_old, thresholds), ensure_ascii=False))

    out = {
        "device": args.device,
        "ranking": args.ranking,
        "n": len(mos),
        "failed": failed[:20],
        "topiq_raw_mos": mos_sum,
        "topiq_old_normalized": old_sum,
        "topiq_raw_histo": _histo(mos, mos_sum["min"], mos_sum["max"]),
        "objective_reversed": _summary(obj10),
        "objective_out_of_range": oob,
        "anchors": {k: (list(v) if v else None) for k, v in anchors.items()},
        "schemes": schemes,
        "baseline_old_final": {"summary": base, "buckets": _bucket_counts(final_old, thresholds)},
        "per_photo": [
            {"filename": rows[i].get("filename"), "mos": round(mos[i], 4)}
            for i in range(len(mos))
        ],
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已保存: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
