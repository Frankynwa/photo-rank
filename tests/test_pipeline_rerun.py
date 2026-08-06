"""分层增量重跑机制单元测试 — rerun_layers 依赖传播 + TOPIQ-only 增量重跑

核心验证点：
1. _expand_rerun_layers 依赖传播正确（l1→全量、l2→l4/topiq/vlm、l4→vlm、topiq/vlm 独立）
2. TOPIQ-only：从 checkpoint 恢复 L1/L2/L4/VLM，只重跑 TOPIQ，VLM 分保留
3. VLM-only：TOPIQ 分保留，只重跑 VLM
"""
import json
from pathlib import Path

import pytest

from core.models import Layer1Result, Layer3Result, Layer4Result, SimilarityCluster
from core.pipeline import _expand_rerun_layers, run_pipeline


class TestExpandRerunLayers:
    """依赖传播规则"""

    def test_none_returns_empty(self):
        assert _expand_rerun_layers(None) == set()
        assert _expand_rerun_layers([]) == set()

    def test_topiq_alone_no_propagation(self):
        """TOPIQ 独立：不牵连其他层"""
        assert _expand_rerun_layers(["topiq"]) == {"topiq"}

    def test_vlm_alone_no_propagation(self):
        """VLM 独立：不牵连其他层"""
        assert _expand_rerun_layers(["vlm"]) == {"vlm"}

    def test_l4_propagates_to_vlm(self):
        """重跑 L4 → VLM 连带（VLM prompt 注入人脸信息）"""
        assert _expand_rerun_layers(["l4"]) == {"l4", "vlm"}

    def test_l2_propagates_all_downstream(self):
        """重跑 L2 → l4/topiq/vlm 连带（代表帧集合变化）"""
        assert _expand_rerun_layers(["l2"]) == {"l2", "l4", "topiq", "vlm"}

    def test_l1_propagates_everything(self):
        """重跑 L1 → 全量连带"""
        assert _expand_rerun_layers(["l1"]) == {"l1", "l2", "l4", "topiq", "vlm"}

    def test_multi_combo(self):
        assert _expand_rerun_layers(["topiq", "vlm"]) == {"topiq", "vlm"}
        assert _expand_rerun_layers(["l1", "topiq"]) == {"l1", "l2", "l4", "topiq", "vlm"}


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """构造完整 checkpoint（completed_layer=4）+ 3 张假照片 + monkeypatch 输出目录"""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    paths = []
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        p = photos_dir / name
        p.write_bytes(b"fake-jpeg")
        paths.append(str(p))

    out_dir = tmp_path / "out"
    platform_dir = out_dir / "xiaohongshu"
    platform_dir.mkdir(parents=True)

    # 完整 v2 checkpoint（L1/L2/L4/L3 全部完成）
    phash_map = {p: f"{i:016x}" for i, p in enumerate(paths)}
    checkpoint = {
        "version": 2,
        "completed_layer": 4,
        "skipped_layers": [],
        "kept": [
            Layer1Result(path=p, filename=Path(p).name, sharpness=100.0, exposure=1.0, noise=0.1, overall=0.7)
            for p in paths
        ],
        "rejected": [],
        "phash_map": phash_map,
        "clusters": [
            SimilarityCluster(
                cluster_id=i, member_count=1, members=[p], representative=p,
                representatives=[p], filename=Path(p).name,
            )
            for i, p in enumerate(paths)
        ],
        "face_results": [
            Layer4Result(path=p, filename=Path(p).name, face_count=0, has_face=False)
            for p in paths
        ],
        # TOPIQ 旧分全是 10.0（缩放 bug 时代），VLM 分有区分度
        "aes_results": [
            Layer3Result(
                path=p, filename=Path(p).name,
                overall_score=10.0, aesthetic_score=10.0, technical_score=10.0, final_score=10.0,
                vlm_overall_score=8.0, composition_score=8.0, color_score=8.0, lighting_score=8.0,
                category="风景",
            )
            for p in paths
        ],
    }
    # 写完整 checkpoint（含各层 model_dump 序列化）
    _write_checkpoint(platform_dir, checkpoint)

    monkeypatch.setattr("core.pipeline.OUTPUT_DIR", out_dir)
    return {
        "photos_dir": str(photos_dir),
        "platform_dir": platform_dir,
        "paths": paths,
    }


def _write_checkpoint(platform_dir: Path, checkpoint: dict) -> None:
    """按各层 model_dump 序列化 checkpoint"""
    data = {
        **checkpoint,
        "kept": [r.model_dump() for r in checkpoint["kept"]],
        "clusters": [c.model_dump() for c in checkpoint["clusters"]],
        "face_results": [r.model_dump() for r in checkpoint["face_results"]],
        "aes_results": [r.model_dump() for r in checkpoint["aes_results"]],
    }
    (platform_dir / "checkpoint.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _guard_imports(monkeypatch):
    """L1/L2/L4/VLM 层函数打桩：若被调用则测试失败（证明不该重跑的层没有重跑）"""
    import core.layer1_objective
    import core.layer2_similarity
    import core.layer3_aesthetic
    import core.layer4_face

    def _fail(*args, **kwargs):
        pytest.fail("该层不应被重跑")

    monkeypatch.setattr(core.layer1_objective, "batch_analyze_with_phash", _fail)
    monkeypatch.setattr(core.layer2_similarity, "cluster_photos", _fail)
    monkeypatch.setattr(core.layer4_face, "analyze_faces", _fail)
    monkeypatch.setattr(core.layer3_aesthetic, "_run_dimension_analysis", _fail)


class TestTopiqOnlyRerun:
    """TOPIQ-only 增量重跑：复用 checkpoint 的 VLM 分，只替换 TOPIQ 分"""

    def test_rerun_topiq_updates_score_keeps_vlm(self, fake_env, monkeypatch):
        import core.layer3_aesthetic

        env = fake_env
        _guard_imports(monkeypatch)

        # TOPIQ 新分：有区分度（修复缩放 bug 后的结果）
        new_scores = {p: s for p, s in zip(env["paths"], [7.0, 6.0, 5.0])}

        def fake_score_topiq(image_paths, device="auto"):
            return [
                Layer3Result(
                    path=str(p), filename=Path(p).name,
                    overall_score=s, aesthetic_score=s, technical_score=s, final_score=s,
                    device=device,
                )
                for p, s in zip(image_paths, [new_scores[str(p)] for p in image_paths])
            ]

        monkeypatch.setattr(core.layer3_aesthetic, "score_topiq", fake_score_topiq)

        result = run_pipeline(photos_dir=env["photos_dir"], platform="xiaohongshu", rerun_layers=["topiq"])

        # 榜单完成且 Top 排序按新 TOPIQ 分
        assert result["ranked"] == 3
        # 读取 full_ranking 验证 TOPIQ 分已更新 + VLM 分保留
        full = json.loads((env["platform_dir"] / "full_ranking.json").read_text(encoding="utf-8"))
        by_path = {r["path"]: r for r in full}
        for p, s in new_scores.items():
            assert by_path[p]["overall_score"] == s, f"{Path(p).name} TOPIQ 分应更新为 {s}"
            assert by_path[p]["vlm_overall_score"] == 8.0, f"{Path(p).name} VLM 分应保留 checkpoint 值"
        # checkpoint 也被更新为新的 TOPIQ 分（下次可继续增量）
        ckpt = json.loads((env["platform_dir"] / "checkpoint.json").read_text(encoding="utf-8"))
        ckpt_by_path = {r["path"]: r for r in ckpt["aes_results"]}
        assert ckpt_by_path[env["paths"][0]]["overall_score"] == 7.0

    def test_rerun_topiq_reorders_ranking(self, fake_env, monkeypatch):
        """新 TOPIQ 分应影响最终排序（v05 从榜首掉到末位）"""
        import core.layer3_aesthetic

        env = fake_env
        _guard_imports(monkeypatch)

        # 让第一张照片 TOPIQ 分大幅下降，应掉出 Top 顺序前列
        scores = {env["paths"][0]: 1.0, env["paths"][1]: 8.0, env["paths"][2]: 8.5}

        def fake_score_topiq(image_paths, device="auto"):
            return [
                Layer3Result(
                    path=str(p), filename=Path(p).name,
                    overall_score=scores[str(p)], aesthetic_score=scores[str(p)],
                    technical_score=scores[str(p)], final_score=scores[str(p)], device=device,
                )
                for p in image_paths
            ]

        monkeypatch.setattr(core.layer3_aesthetic, "score_topiq", fake_score_topiq)
        run_pipeline(photos_dir=env["photos_dir"], platform="xiaohongshu", rerun_layers=["topiq"])

        full = json.loads((env["platform_dir"] / "full_ranking.json").read_text(encoding="utf-8"))
        assert full[0]["path"] == env["paths"][2]  # 8.5 分居首
        assert full[-1]["path"] == env["paths"][0]  # 1.0 分垫底


class TestVlmOnlyRerun:
    """VLM-only 增量重跑：TOPIQ 分保留，只重跑 VLM"""

    def test_rerun_vlm_keeps_topiq_updates_vlm(self, fake_env, monkeypatch):
        import core.layer3_aesthetic

        env = fake_env

        # 除 VLM 外所有层打桩：不该重跑
        import core.layer1_objective
        import core.layer2_similarity
        import core.layer4_face

        def _fail(*args, **kwargs):
            pytest.fail("该层不应被重跑")

        monkeypatch.setattr(core.layer1_objective, "batch_analyze_with_phash", _fail)
        monkeypatch.setattr(core.layer2_similarity, "cluster_photos", _fail)
        monkeypatch.setattr(core.layer4_face, "analyze_faces", _fail)

        # TOPIQ 保留 checkpoint 分（全 10.0）；VLM 重新评分（区分度分）
        def fake_dim_analysis(image_path, face_summary=None):
            return {"composition_score": 6.0, "color_score": 6.5, "lighting_score": 6.0,
                    "overall_aesthetic_score": 6.2, "category": "风景"}

        monkeypatch.setattr(core.layer3_aesthetic, "_run_dimension_analysis", fake_dim_analysis)
        # VLM-only 不该重跑 TOPIQ：若被调用则失败
        monkeypatch.setattr(core.layer3_aesthetic, "score_topiq",
                            lambda *a, **k: pytest.fail("VLM-only 不应重跑 TOPIQ"))

        # 需要 API key 才会走 VLM 路径
        monkeypatch.setenv("QWEN_API_KEY", "test-key")

        result = run_pipeline(photos_dir=env["photos_dir"], platform="xiaohongshu", rerun_layers=["vlm"])
        assert result["ranked"] == 3

        full = json.loads((env["platform_dir"] / "full_ranking.json").read_text(encoding="utf-8"))
        for r in full:
            # TOPIQ 分保留 checkpoint 的 10.0（归一化后仍 10）
            assert r["overall_score"] == 10.0
            # VLM 分更新为新评分
            assert r["vlm_overall_score"] == pytest.approx(6.2)
