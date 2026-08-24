"""黄金回归集：钳制误伤 bug 的离线防回归断言（mock VLM 响应，不消耗 API）

历史 bug 场景固化（2026-08 奶泡误检事件）：
- 饮品照（无人脸主体，奶泡被误检为脸）被钳制后归一化压到 2.63
- 墨镜/侧脸照片因 clarity 失真被误钳制
- 先钳制后归一化的二次惩罚
每次改 prompt/钳制/检测逻辑后跑本文件，防止同类 bug 回归。
"""
from core.layer3_aesthetic import (
    _apply_dims_to_result,
    clamp_hard_flaws_after_normalize,
)
from core.layer4_face import _is_plausible_face
from core.models import Layer3Result


def _dims(subject=None, flaws=None, overall=8.2):
    d = {"composition_score": 8.5, "color_score": 8.0, "lighting_score": 8.0,
         "overall_aesthetic_score": overall, "reason": "golden case"}
    if subject is not None:
        d["person_is_subject"] = subject
    if flaws is not None:
        d["hard_flaws"] = flaws
    return d


class TestGoldenClampCases:
    """四类历史误伤/真伤场景的钳制判定固化"""

    def test_drink_photo_fake_face_never_clamps(self):
        """饮品照：VLM 裁断人物非主体（误检假脸/背景路人）→ 即使 L4 blur 触发也永不钳制"""
        r = Layer3Result(path="/drink.jpg", filename="drink.jpg")
        _apply_dims_to_result(r, _dims(subject=False, flaws=[]), hard_flag="blur")
        assert r.vlm_clamp is False
        clamp_hard_flaws_after_normalize([r])
        assert r.vlm_overall_score == 8.2  # 分数完整保留

    def test_sunglasses_no_clamp(self):
        """墨镜照：L4 clarity 失真触发 blur，但 VLM 未观察到人脸模糊 → 不钳制"""
        r = Layer3Result(path="/sun.jpg", filename="sun.jpg")
        _apply_dims_to_result(r, _dims(subject=True, flaws=[]), hard_flag="blur")
        assert r.vlm_clamp is False
        clamp_hard_flaws_after_normalize([r])
        assert r.vlm_overall_score == 8.2

    def test_profile_face_no_clamp(self):
        """侧脸照：同墨镜——blur 触发但无 VLM 佐证 → 不钳制"""
        r = Layer3Result(path="/profile.jpg", filename="profile.jpg")
        _apply_dims_to_result(r, _dims(subject=True, flaws=[]), hard_flag="blur")
        assert r.vlm_clamp is False

    def test_true_blur_still_clamps(self):
        """真糊照：blur 触发 + VLM 确认主体且自报人脸模糊 → 钳制到 6"""
        r = Layer3Result(path="/blur.jpg", filename="blur.jpg")
        _apply_dims_to_result(r, _dims(subject=True, flaws=["人脸模糊"]), hard_flag="blur")
        assert r.vlm_clamp is True
        clamp_hard_flaws_after_normalize([r])
        assert r.vlm_overall_score == 6.0

    def test_subject_blink_still_clamps(self):
        """主体闭眼（blink 路径）→ 钳制到 6，不受 F2 双证据影响"""
        r = Layer3Result(path="/blink.jpg", filename="blink.jpg")
        _apply_dims_to_result(r, _dims(subject=True, flaws=[]), hard_flag="blink")
        assert r.vlm_clamp is True
        clamp_hard_flaws_after_normalize([r])
        assert r.vlm_overall_score == 6.0

    def test_no_double_punishment(self):
        """F3 回归：被钳制照片原始分正常参与归一化，归一化后才钳 ≤6，
        杜绝"先钳到 6 再被 p5-p95 拉到 2.63"的二次惩罚"""
        results = [
            Layer3Result(path=f"/x/{i}", filename=f"{i}.jpg", vlm_overall_score=s)
            for i, s in enumerate([5.0, 6.0, 7.0, 8.0, 9.0], 1)
        ]
        # 第 3 张（原始 7 分）被判定人像硬伤
        results[2].vlm_clamp = True
        from core.layer3_aesthetic import _normalize_vlm_batch
        _normalize_vlm_batch(results)
        # 原始分正常参与归一化：7 分被拉伸到 ~6.67（p5=5/p95=8），
        # 而不是旧 bug 的"先钳到 6 再被拉伸压到 2.63"
        assert abs(results[2].vlm_overall_score - 6.6667) < 1e-3
        clamp_hard_flaws_after_normalize(results)
        assert results[2].vlm_overall_score == 6.0
        # 其余照片不受钳制影响
        assert results[0].vlm_overall_score == 0.1
        assert results[4].vlm_overall_score == 10.0


class TestGoldenFaceGate:
    """降级路径融合门控：person_is_subject=False 时 face 权重归零"""

    def test_fake_face_not_polluting_fusion(self):
        """VLM 未评分（降级路径）时，假脸的低分人脸分不应污染融合分"""
        from core.pipeline import _compute_final_score
        weights = {"composition": 0.35, "color": 0.30, "lighting": 0.20, "face": 0.15}
        dim_w_sum = 0.85
        face_scores = {"/drink.jpg": 0.23}  # 假脸的低分
        l1_scores = {"/drink.jpg": 0.5}

        r_gate = Layer3Result(path="/drink.jpg", filename="drink.jpg",
                              person_is_subject=False, overall_score=6.0)
        r_open = Layer3Result(path="/drink.jpg", filename="drink.jpg",
                              overall_score=6.0)
        s_gate = _compute_final_score(r_gate, weights, dim_w_sum, face_scores, l1_scores)
        s_open = _compute_final_score(r_open, weights, dim_w_sum, face_scores, l1_scores)
        s_noface = _compute_final_score(r_open, weights, dim_w_sum, {}, l1_scores)
        # 门控后与"无人脸"完全等价，且高于未门控（假脸 0.23 拖分被消除）
        assert s_gate == s_noface
        assert s_gate > s_open


# ---------------------------------------------------------------------------
# 假脸几何校验回归（pareidolia 拦截 + 真脸不误杀）
# ---------------------------------------------------------------------------

class _Pt:
    def __init__(self, x, y):
        self.x, self.y = x, y


def _make_lm(eye_y=0.35, nose_y=0.50, mouth_y=0.65, y_min=0.2, y_max=0.9):
    """构造 478 点假 landmark：默认点铺满 bbox，关键五官索引单独定位"""
    lm = [_Pt(0.5, (y_min + y_max) / 2) for _ in range(478)]
    # 边界点撑出 bbox（bw=0.6, bh=0.7，高宽比 1.17）
    lm[10] = _Pt(0.5, y_min)   # 额头
    lm[152] = _Pt(0.5, y_max)  # 下巴
    lm[234] = _Pt(0.2, 0.55)   # 右脸颊边
    lm[454] = _Pt(0.8, 0.55)   # 左脸颊边
    # 双眼（眼距 0.20，占 bw 33%）
    lm[33], lm[133] = _Pt(0.35, eye_y), _Pt(0.45, eye_y)
    lm[362], lm[263] = _Pt(0.55, eye_y), _Pt(0.65, eye_y)
    lm[4] = _Pt(0.5, nose_y)                      # 鼻尖
    lm[61], lm[291] = _Pt(0.42, mouth_y), _Pt(0.58, mouth_y)  # 嘴角（嘴宽 0.16）
    return lm


class TestPareidoliaGeometry:
    def test_real_face_layout_passes(self):
        """正常五官拓扑（眼<鼻<嘴）应通过校验"""
        assert _is_plausible_face(_make_lm()) is True

    def test_scrambled_order_rejected(self):
        """五官垂直顺序错乱（奶泡类纹理误检特征）应被拦截"""
        assert _is_plausible_face(_make_lm(nose_y=0.70, mouth_y=0.45)) is False

    def test_eyes_too_close_rejected(self):
        """双眼过挤（瞳距 < bbox 宽 20%）应被拦截"""
        lm = _make_lm()
        lm[33], lm[133] = _Pt(0.48, 0.35), _Pt(0.50, 0.35)
        lm[362], lm[263] = _Pt(0.50, 0.35), _Pt(0.52, 0.35)
        assert _is_plausible_face(lm) is False

    def test_flat_texture_rejected(self):
        """极端扁平 bbox（高宽比 <0.75，织物/云影类纹理）应被拦截"""
        assert _is_plausible_face(
            _make_lm(eye_y=0.60, nose_y=0.70, mouth_y=0.80, y_min=0.45, y_max=0.9)
        ) is False
