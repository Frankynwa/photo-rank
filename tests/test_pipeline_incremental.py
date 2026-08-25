"""新增照片增量模式单元测试 — incremental=True 时只处理新增照片，旧照片复用 checkpoint

核心验证点：
1. 无新增照片：incremental 全部复用 checkpoint（各层不重跑）
2. 有新增照片：L1 只算新照片、L2 重聚类、L4/TOPIQ/VLM 只对新增代表评分
3. 增量后 checkpoint 更新（新照片不再是"新增"，旧代表评分保留）
"""
import json
from pathlib import Path

import imagehash
import numpy as np
import pytest

from config import QWEN_MODEL
from core.layer3_aesthetic import vlm_prompt_fingerprint
from core.models import Layer1Result, Layer3Result, Layer4Result, SimilarityCluster
from core.pipeline import run_pipeline


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """完整 checkpoint（completed_layer=4）+ 3 张旧照片 + monkeypatch 输出目录"""
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

    phash_map = {p: f"{i:016x}" for i, p in enumerate(paths)}
    checkpoint = {
        "version": 3,
        "completed_layer": 4,
        "skipped_layers": [],
        # 新鲜度指纹：与当前 prompt/模型一致 → 旧分可复用（不一致场景见 TestCheckpointStale）
        "meta": {"prompt_hash": vlm_prompt_fingerprint(), "model": QWEN_MODEL},
        "kept": [
            Layer1Result(path=p, filename=Path(p).name, sharpness=300.0, exposure=1.0, noise=0.05, overall=0.8)
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
        "aes_results": [
            Layer3Result(
                path=p, filename=Path(p).name,
                overall_score=6.0, aesthetic_score=6.0, technical_score=6.0, final_score=6.0,
                vlm_overall_score=8.0, composition_score=8.0, color_score=8.0, lighting_score=8.0,
                category="风景",
            )
            for p in paths
        ],
    }
    data = {
        **checkpoint,
        "kept": [r.model_dump() for r in checkpoint["kept"]],
        "clusters": [c.model_dump() for c in checkpoint["clusters"]],
        "face_results": [r.model_dump() for r in checkpoint["face_results"]],
        "aes_results": [r.model_dump() for r in checkpoint["aes_results"]],
    }
    (platform_dir / "checkpoint.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8")

    monkeypatch.setattr("core.pipeline.OUTPUT_DIR", out_dir)
    return {
        "photos_dir": str(photos_dir),
        "platform_dir": platform_dir,
        "paths": paths,
    }


def _add_photo(photos_dir, name="d.jpg"):
    p = Path(photos_dir) / name
    p.write_bytes(b"fake-jpeg")
    return str(p)


def _install_fakes(monkeypatch, new_paths):
    """安装各层 fake，记录调用参数；返回调用记录 dict"""
    import core.layer1_objective
    import core.layer2_similarity
    import core.layer3_aesthetic
    import core.layer4_face

    calls = {"l1": [], "l2": [], "l4": [], "topiq": [], "vlm": []}

    def fake_l1(image_paths):
        calls["l1"].append([str(p) for p in image_paths])
        results, phash = [], {}
        for i, p in enumerate(image_paths):
            results.append(Layer1Result(
                path=str(p), filename=Path(p).name,
                sharpness=200.0, exposure=1.0, noise=0.05, overall=0.8,
            ))
            # 直接构造 ImageHash（避免 hex_to_hash 的版本差异）
            bits = np.zeros((8, 8), dtype=bool)
            bits[0, i % 8] = True
            phash[str(p)] = imagehash.ImageHash(bits)
        return results, phash

    def fake_cluster(paths, sharpness_scores, precomputed_hashes=None, same_date=False):
        calls["l2"].append([str(p) for p in paths])
        return [
            SimilarityCluster(
                cluster_id=i, member_count=1, members=[str(p)], representative=str(p),
                representatives=[str(p)], filename=Path(p).name,
            )
            for i, p in enumerate(paths)
        ]

    def fake_faces(image_paths, on_photo_done=None):
        calls["l4"].append([str(p) for p in image_paths])
        out = []
        for p in image_paths:
            r = Layer4Result(path=str(p), filename=Path(p).name, face_count=0, has_face=False)
            out.append(r)
            if on_photo_done is not None:
                on_photo_done(r)
        return out

    def fake_topiq(image_paths, device="auto"):
        calls["topiq"].append([str(p) for p in image_paths])
        return [
            Layer3Result(
                path=str(p), filename=Path(p).name,
                overall_score=7.0, aesthetic_score=7.0, technical_score=7.0, final_score=7.0,
                device=device,
            )
            for p in image_paths
        ]

    def fake_vlm(image_path, face_summary=None):
        calls["vlm"].append(str(image_path))
        return {"composition_score": 6.0, "color_score": 6.5, "lighting_score": 6.0,
                "overall_aesthetic_score": 6.2, "category": "风景"}

    monkeypatch.setattr(core.layer1_objective, "batch_analyze_with_phash", fake_l1)
    monkeypatch.setattr(core.layer2_similarity, "cluster_photos", fake_cluster)
    monkeypatch.setattr(core.layer4_face, "analyze_faces", fake_faces)
    monkeypatch.setattr(core.layer4_face, "to_face_prompt_summary", lambda r: "")
    monkeypatch.setattr(core.layer3_aesthetic, "score_topiq", fake_topiq)
    monkeypatch.setattr(core.layer3_aesthetic, "_run_dimension_analysis", fake_vlm)
    return calls


class TestIncrementalNoNewPhotos:
    """无新增照片：incremental 全部复用 checkpoint，任何层都不重跑"""

    def test_no_new_photos_reuses_everything(self, fake_env, monkeypatch):
        env = fake_env
        import core.layer1_objective
        import core.layer2_similarity
        import core.layer3_aesthetic
        import core.layer4_face

        def _fail(*args, **kwargs):
            pytest.fail("无新增照片时不应重跑任何层")

        monkeypatch.setattr(core.layer1_objective, "batch_analyze_with_phash", _fail)
        monkeypatch.setattr(core.layer2_similarity, "cluster_photos", _fail)
        monkeypatch.setattr(core.layer4_face, "analyze_faces", _fail)
        monkeypatch.setattr(core.layer3_aesthetic, "score_topiq", _fail)
        monkeypatch.setattr(core.layer3_aesthetic, "_run_dimension_analysis", _fail)

        result = run_pipeline(photos_dir=env["photos_dir"], platform="xiaohongshu", incremental=True)

        assert result["ranked"] == 3
        full = json.loads((env["platform_dir"] / "full_ranking.json").read_text(encoding="utf-8"))
        assert len(full) == 3
        # 旧代表 VLM 分保留 checkpoint 值
        assert all(r["vlm_overall_score"] == 8.0 for r in full)


class TestIncrementalWithNewPhoto:
    """有新增照片：只对新照片跑 L1，只对新增代表跑 L4/TOPIQ/VLM"""

    def test_only_new_photo_scored(self, fake_env, monkeypatch):
        env = fake_env
        new_path = _add_photo(env["photos_dir"], "d.jpg")
        monkeypatch.setenv("QWEN_API_KEY", "test-key")

        calls = _install_fakes(monkeypatch, [new_path])

        result = run_pipeline(photos_dir=env["photos_dir"], platform="xiaohongshu", incremental=True)

        assert result["ranked"] == 4, "增量后应有 4 个代表进入排名"

        # L1 只算新增照片
        assert len(calls["l1"]) == 1 and calls["l1"][0] == [new_path], "L1 应只处理新增照片"
        # L2 全量重聚类（旧+新）
        assert len(calls["l2"]) == 1 and len(calls["l2"][0]) == 4, "L2 应全量聚类 4 张"
        # L4/TOPIQ 只对新代表
        assert calls["l4"] == [[new_path]], "L4 应只评分新增代表"
        assert calls["topiq"] == [[new_path]], "TOPIQ 应只评分新增代表"
        # VLM 只对新代表
        assert calls["vlm"] == [new_path], "VLM 应只评分新增代表"

        # 结果合并：VLM 全量统一重归一化（raw 8.0×3 + 6.2 → p5-p95 拉伸）：
        # 旧 8.0 → 10.0，新 6.2 → 正下限 0.1；原始分经 vlm_raw_score 保留
        full = json.loads((env["platform_dir"] / "full_ranking.json").read_text(encoding="utf-8"))
        by_path = {r["path"]: r for r in full}
        assert len(full) == 4
        for p in env["paths"]:
            assert by_path[p]["vlm_overall_score"] == 10.0, f"{Path(p).name} 应统一重归一化"
            assert by_path[p]["vlm_raw_score"] == 8.0, "归一化前原始分应保留"
        assert by_path[new_path]["vlm_raw_score"] == pytest.approx(6.2), "新照片原始分应保留"
        assert by_path[new_path]["vlm_overall_score"] == pytest.approx(0.1), "新照片处统一标尺下限"
        # 新照片 TOPIQ 分 7.0（旧记录缺 raw → TOPIQ 重归一化不触发，保持原值）
        assert by_path[new_path]["overall_score"] == 7.0

        # checkpoint 已更新：新照片进入，下次不再是"新增"
        ckpt = json.loads((env["platform_dir"] / "checkpoint.json").read_text(encoding="utf-8"))
        ckpt_paths = {r["path"] for r in ckpt["kept"]}
        assert new_path in ckpt_paths, "新照片应写入 checkpoint kept"

    def test_second_incremental_no_work(self, fake_env, monkeypatch):
        """增量跑完后再次 incremental（无新增）→ 各层不重跑"""
        env = fake_env
        new_path = _add_photo(env["photos_dir"], "d.jpg")
        monkeypatch.setenv("QWEN_API_KEY", "test-key")
        _install_fakes(monkeypatch, [new_path])
        run_pipeline(photos_dir=env["photos_dir"], platform="xiaohongshu", incremental=True)

        # 第二次：无新增 → 全部复用
        import core.layer1_objective
        import core.layer2_similarity
        import core.layer3_aesthetic
        import core.layer4_face

        def _fail(*args, **kwargs):
            pytest.fail("第二次增量（无新增）不应重跑任何层")

        monkeypatch.setattr(core.layer1_objective, "batch_analyze_with_phash", _fail)
        monkeypatch.setattr(core.layer2_similarity, "cluster_photos", _fail)
        monkeypatch.setattr(core.layer4_face, "analyze_faces", _fail)
        monkeypatch.setattr(core.layer3_aesthetic, "score_topiq", _fail)
        monkeypatch.setattr(core.layer3_aesthetic, "_run_dimension_analysis", _fail)

        result = run_pipeline(photos_dir=env["photos_dir"], platform="xiaohongshu", incremental=True)
        assert result["ranked"] == 4


class TestCheckpointStale:
    """VLM 分新鲜度校验：prompt/模型指纹不匹配（或 v2 无指纹）→ 旧 VLM 分作废全量重评"""

    def _stale_run(self, fake_env, monkeypatch, mutate_meta):
        env = fake_env
        ckpt_file = env["platform_dir"] / "checkpoint.json"
        ckpt = json.loads(ckpt_file.read_text(encoding="utf-8"))
        mutate_meta(ckpt)
        ckpt_file.write_text(json.dumps(ckpt, ensure_ascii=False), encoding="utf-8")
        monkeypatch.setenv("QWEN_API_KEY", "test-key")
        calls = _install_fakes(monkeypatch, [])
        result = run_pipeline(photos_dir=env["photos_dir"], platform="xiaohongshu", incremental=True)
        return env, calls, result

    def test_v2_no_meta_invalidates_vlm(self, fake_env, monkeypatch):
        """v2 旧 checkpoint 无指纹 → 全部 3 张重评 VLM，TOPIQ/L4 复用"""

        def _to_v2(ckpt):
            ckpt["version"] = 2
            ckpt.pop("meta", None)

        env, calls, result = self._stale_run(fake_env, monkeypatch, _to_v2)
        assert result["ranked"] == 3
        assert sorted(calls["vlm"]) == sorted(env["paths"]), "旧 VLM 分应全量重评"
        assert calls["l4"] == [], "L4 应复用 checkpoint"
        assert calls["topiq"] == [], "TOPIQ 应复用 checkpoint"

    def test_hash_mismatch_invalidates_vlm(self, fake_env, monkeypatch):
        """指纹不匹配（prompt 已改）→ 同样全量重评 VLM"""
        env, calls, result = self._stale_run(
            fake_env, monkeypatch,
            lambda ckpt: ckpt["meta"].update({"prompt_hash": "deadbeef00000000"})
        )
        assert result["ranked"] == 3
        assert sorted(calls["vlm"]) == sorted(env["paths"])

    def test_rerun_meta_written_after_stale(self, fake_env, monkeypatch):
        """过期重评后产物携带新指纹：checkpoint meta 与 run_meta.json 一致"""
        env, calls, result = self._stale_run(
            fake_env, monkeypatch, lambda ckpt: ckpt.pop("meta", None)
        )
        ckpt = json.loads((env["platform_dir"] / "checkpoint.json").read_text(encoding="utf-8"))
        assert ckpt["version"] == 3
        assert ckpt["meta"]["prompt_hash"] == vlm_prompt_fingerprint()
        meta = json.loads((env["platform_dir"] / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["prompt_hash"] == ckpt["meta"]["prompt_hash"]
        assert meta["model"] == QWEN_MODEL
