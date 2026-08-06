"""Layer 2 相似聚类单元测试"""
from pathlib import Path

import numpy as np
from PIL import Image

from core.layer2_similarity import (
    _UnionFind,
    cluster_photos,
    compute_phash,
    photo_date_key,
)

# ---------------------------------------------------------------------------
# _UnionFind 数据结构测试
# ---------------------------------------------------------------------------

class TestUnionFind:
    def test_initial_state(self):
        uf = _UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_connects_elements(self):
        uf = _UnionFind(4)
        uf.union(0, 1)
        assert uf.find(0) == uf.find(1)

    def test_transitive_union(self):
        uf = _UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(2)

    def test_disjoint_elements(self):
        uf = _UnionFind(4)
        uf.union(0, 1)
        uf.union(2, 3)
        assert uf.find(0) != uf.find(2)

    def test_union_same_element(self):
        uf = _UnionFind(3)
        uf.union(1, 1)
        assert uf.find(1) == 1


# ---------------------------------------------------------------------------
# compute_phash 测试（需要真实图片文件）
# ---------------------------------------------------------------------------

def _make_solid_image(path: Path, color=(128, 128, 128), size=(64, 64)):
    img = Image.new("RGB", size, color)
    img.save(str(path))
    return path


def _make_gradient_image(path: Path, size=(64, 64)):
    """生成渐变图片，比纯色包含更多信息"""
    arr = np.tile(np.linspace(0, 255, size[0], dtype=np.uint8), (size[1], 1))
    img = Image.fromarray(arr, mode="L").convert("RGB")
    img.save(str(path))
    return path


class TestComputePhash:
    def test_valid_image_returns_hash(self, tmp_path: Path):
        path = _make_solid_image(tmp_path / "solid.png")
        h = compute_phash(path)
        assert h is not None

    def test_nonexistent_returns_none(self, tmp_path: Path):
        h = compute_phash(tmp_path / "missing.png")
        assert h is None

    def test_same_image_same_hash(self, tmp_path: Path):
        """同一张图片的哈希应相同"""
        path = _make_gradient_image(tmp_path / "grad.png")
        h1 = compute_phash(path)
        h2 = compute_phash(path)
        assert h1 == h2


# ---------------------------------------------------------------------------
# cluster_photos 集成测试
# ---------------------------------------------------------------------------

class TestClusterPhotos:
    def test_identical_images_clustered(self, tmp_path: Path):
        """完全相同的图片应被聚到同一簇"""
        paths = []
        for i in range(3):
            p = _make_gradient_image(tmp_path / f"same_{i}.png")
            paths.append(str(p))

        sharpness = {p: 100.0 for p in paths}
        clusters = cluster_photos(paths, sharpness, threshold=10)

        # 所有相同图片应在同一簇
        assert len(clusters) == 1
        assert clusters[0].member_count == 3

    def test_different_images_separate_clusters(self, tmp_path: Path):
        """内容完全不同的图片应在不同簇"""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        paths = []
        for i, c in enumerate(colors):
            # 使用较大尺寸 + 不同颜色确保哈希差异足够大
            arr = np.zeros((128, 128, 3), dtype=np.uint8)
            arr[:, :] = c
            # 加入显著纹理让哈希更有区分度
            for row in range(0, 128, 4):
                arr[row, :] = (255 - c[0], 255 - c[1], 255 - c[2])
            p = tmp_path / f"diff_{i}.png"
            Image.fromarray(arr).save(str(p))
            paths.append(str(p))

        sharpness = {p: 50.0 for p in paths}
        # 使用很小的阈值确保不同图片不被聚合
        clusters = cluster_photos(paths, sharpness, threshold=2)

        assert len(clusters) >= 2, "完全不同图片应有多个簇"

    def test_sharpness_selects_representative(self, tmp_path: Path):
        """聚类中清晰度最高的图片应被选为代表"""
        paths = []
        for i in range(3):
            p = _make_gradient_image(tmp_path / f"rep_{i}.png")
            paths.append(str(p))

        # 第二张清晰度最高
        sharpness = {paths[0]: 10.0, paths[1]: 999.0, paths[2]: 20.0}
        clusters = cluster_photos(paths, sharpness, threshold=10)

        assert len(clusters) == 1
        assert clusters[0].representative == paths[1]

    def test_representatives_top2_by_sharpness(self, tmp_path: Path):
        """每组应按清晰度取前 2 张作为代表，最高清晰度排首位"""
        paths = []
        for i in range(4):
            p = _make_gradient_image(tmp_path / f"top2_{i}.png")
            paths.append(str(p))

        # 清晰度：p1 > p3 > p2 > p0
        sharpness = {paths[0]: 10.0, paths[1]: 999.0, paths[2]: 20.0, paths[3]: 500.0}
        clusters = cluster_photos(paths, sharpness, threshold=10)

        assert len(clusters) == 1
        assert clusters[0].representatives == [paths[1], paths[3]]
        assert clusters[0].representative == paths[1]

    def test_single_member_representatives(self, tmp_path: Path):
        """单成员簇的 representatives 应只含自身"""
        p = _make_gradient_image(tmp_path / "solo.png")
        clusters = cluster_photos([str(p)], {str(p): 50.0}, threshold=10)
        assert len(clusters) == 1
        assert clusters[0].representatives == [str(p)]

    def test_empty_input(self):
        """空输入应返回空列表"""
        clusters = cluster_photos([], {}, threshold=10)
        assert clusters == []

    def test_single_image(self, tmp_path: Path):
        """单张图片应产生一个簇"""
        p = _make_gradient_image(tmp_path / "single.png")
        clusters = cluster_photos([str(p)], {str(p): 50.0}, threshold=10)
        assert len(clusters) == 1
        assert clusters[0].member_count == 1


class TestPhotoDateKey:
    """文件名拍摄日期提取（同日期约束的基础）"""

    def test_standard_photo_name(self):
        assert photo_date_key("uploads/MVIMG_20260707_064715.jpg") == "20260707"

    def test_heic_photo_name(self):
        assert photo_date_key("uploads/IMG_20260713_105006.HEIC") == "20260713"

    def test_video_frame_no_date(self):
        """视频帧文件名无日期 → 返回 None（不参与同日期约束）"""
        assert photo_date_key("uploads/video_frames/v01_ab12cd_t10.5.jpg") is None

    def test_missing_date_returns_none(self):
        assert photo_date_key("some/random_name.png") is None


class TestClusterPhotosSameDate:
    """同日期约束：跨日期 pHash 相近的照片不应合并"""

    def test_cross_date_not_merged_when_same_date(self, tmp_path: Path):
        """同一张图复制到不同日期文件名，same_date=True 时不应合并"""
        # 两张内容完全相同的图（pHash 距离 0），但文件名日期不同
        p1 = _make_gradient_image(tmp_path / "MVIMG_20260707_100000.jpg")
        p2 = _make_gradient_image(tmp_path / "MVIMG_20260708_100000.jpg")
        paths = [str(p1), str(p2)]
        sharpness = {p: 100.0 for p in paths}

        # 无约束：完全相同必然合并
        clusters_loose = cluster_photos(paths, sharpness, threshold=10, same_date=False)
        assert len(clusters_loose) == 1

        # 同日期约束：跨日期不合并
        clusters_strict = cluster_photos(paths, sharpness, threshold=10, same_date=True)
        assert len(clusters_strict) == 2

    def test_same_date_still_merged(self, tmp_path: Path):
        """同日期内完全相同照片仍应合并（连拍去重不受影响）"""
        p1 = _make_gradient_image(tmp_path / "MVIMG_20260707_100000.jpg")
        p2 = _make_gradient_image(tmp_path / "MVIMG_20260707_100002.jpg")
        paths = [str(p1), str(p2)]
        sharpness = {p: 100.0 for p in paths}

        clusters = cluster_photos(paths, sharpness, threshold=10, same_date=True)
        assert len(clusters) == 1
        assert clusters[0].member_count == 2
