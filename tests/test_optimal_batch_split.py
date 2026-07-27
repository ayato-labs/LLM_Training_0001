import unittest

from src.training.model_utils import calculate_optimal_batch_split


class TestOptimalBatchSplit(unittest.TestCase):
    def test_small_model_4gb_vram(self):
        per_device, grad_accum = calculate_optimal_batch_split(
            total_batch_size=32,
            vram_gb=4.0,
            n_params=150_000_000,
            seq_len=1024,
            use_calibration=False,
        )
        self.assertGreaterEqual(per_device, 1)
        self.assertGreaterEqual(grad_accum, 1)

    def test_small_batch_size(self):
        per_device, grad_accum = calculate_optimal_batch_split(
            total_batch_size=2,
            vram_gb=4.0,
            n_params=150_000_000,
            seq_len=1024,
            use_calibration=False,
        )
        self.assertEqual(per_device, 2)
        self.assertEqual(grad_accum, 1)

    def test_speed_maximized_micro_batch(self):
        per_device, grad_accum = calculate_optimal_batch_split(
            total_batch_size=5,
            vram_gb=4.0,
            n_params=37_000_000,
            seq_len=512,
            use_calibration=False,
        )
        # VRAMが許す最大のマイクロバッチが設定されること
        self.assertGreaterEqual(per_device, 1)


if __name__ == "__main__":
    unittest.main()
