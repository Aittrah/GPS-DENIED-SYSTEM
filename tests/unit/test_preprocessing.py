import numpy as np
import pytest
import cv2
from vns.preprocessing.satellite_preprocessor import SatellitePreprocessor, TARGET_SIZE
from vns.preprocessing.uav_preprocessor import UAVPreprocessor
from vns.preprocessing.patch_generator import PatchGenerator, PATCH_SIZE, STRIDE
from vns.core.geo_utils import NEDPoint

def make_image(h=512, w=512):
    return (np.random.rand(h, w, 3) * 255).astype(np.uint8)

def make_checkerboard(size=512):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    tile = 32
    for i in range(0, size, tile):
        for j in range(0, size, tile):
            if (i // tile + j // tile) % 2 == 0:
                img[i:i+tile, j:j+tile] = 255
    return img

def center():
    return NEDPoint(0.0, 0.0, -50.0)

class TestSatellitePreprocessor:

    def test_output_is_target_size(self):
        proc = SatellitePreprocessor()
        result = proc.load_and_preprocess_array(make_image(1024, 1024))
        assert result.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3)

    def test_output_is_uint8(self):
        proc = SatellitePreprocessor()
        result = proc.load_and_preprocess_array(make_image())
        assert result.dtype == np.uint8

    def test_load_missing_file_returns_none(self):
        proc = SatellitePreprocessor()
        assert proc.load_and_preprocess('nonexistent.jpg') is None

    def test_small_image_upscaled(self):
        proc = SatellitePreprocessor()
        result = proc.load_and_preprocess_array(make_image(64, 64))
        assert result.shape[:2] == (TARGET_SIZE[1], TARGET_SIZE[0])

    def test_output_not_all_zeros(self):
        proc = SatellitePreprocessor()
        result = proc.load_and_preprocess_array(make_checkerboard())
        assert result.sum() > 0

class TestUAVPreprocessor:

    def test_output_is_target_size(self):
        proc = UAVPreprocessor()
        result = proc.preprocess_frame(make_image())
        assert result is not None
        assert result.shape == (TARGET_SIZE[1], TARGET_SIZE[0], 3)

    def test_empty_frame_returns_none(self):
        proc = UAVPreprocessor()
        assert proc.preprocess_frame(np.array([])) is None

    def test_target_size_matches_satellite(self):
        assert UAVPreprocessor.TARGET_SIZE == TARGET_SIZE

    def test_output_dtype_is_uint8(self):
        proc = UAVPreprocessor()
        result = proc.preprocess_frame(make_image())
        assert result is not None
        assert result.dtype == np.uint8

class TestPatchGenerator:

    def test_patch_shape_is_correct(self):
        gen = PatchGenerator()
        patches = gen.generate_patches(make_image(), center(), 0.5, 'satellite', 'test')
        for p in patches:
            assert p.data.shape == (PATCH_SIZE, PATCH_SIZE, 3)

    def test_patches_have_unique_ids(self):
        gen = PatchGenerator()
        patches = gen.generate_patches(make_image(), center(), 0.5, 'satellite', 'img1')
        ids = [p.patch_id for p in patches]
        assert len(ids) == len(set(ids))

    def test_correct_patch_count(self):
        gen = PatchGenerator()
        patches = gen.generate_patches(make_image(512, 512), center(), 0.5, 'uav', 'test')
        expected = ((512 - PATCH_SIZE) // STRIDE + 1) ** 2
        assert len(patches) == expected

    def test_source_label_set_correctly(self):
        gen = PatchGenerator()
        patches = gen.generate_patches(make_image(), center(), 0.5, 'satellite', 't')
        assert all(p.source == 'satellite' for p in patches)

    def test_patch_positions_differ(self):
        gen = PatchGenerator()
        patches = gen.generate_patches(make_image(), center(), 0.5, 'uav', 't')
        positions = [(p.center_position.north, p.center_position.east) for p in patches]
        assert len(positions) == len(set(positions))

    def test_single_patch_resize(self):
        gen = PatchGenerator()
        patch = gen.generate_single_patch(make_image(512, 512), center(), 'uav', 'live')
        assert patch.data.shape == (PATCH_SIZE, PATCH_SIZE, 3)

    def test_custom_patch_size_uses_half_patch_stride_by_default(self):
        gen = PatchGenerator(patch_size=128)
        patches = gen.generate_patches(make_image(512, 512), center(), 0.5, 'uav', 'custom')
        expected = ((512 - 128) // 64 + 1) ** 2
        assert len(patches) == expected
        assert all(p.data.shape == (128, 128, 3) for p in patches)
        assert gen.stride == 64

    def test_patch_offsets_track_parent_image_coordinates(self):
        gen = PatchGenerator()
        patches = gen.generate_patches(make_image(512, 512), center(), 0.5, 'uav', 'coords')
        assert patches[0].offset_x == 0
        assert patches[0].offset_y == 0
        assert patches[1].offset_x == STRIDE
        assert patches[1].offset_y == 0

    def test_patch_data_is_copy(self):
        gen = PatchGenerator()
        img = make_image()
        original_copy = img.copy()
        patches = gen.generate_patches(img, center(), 0.5, 'uav', 't')
        patches[0].data[:] = 0
        assert np.array_equal(img, original_copy)
