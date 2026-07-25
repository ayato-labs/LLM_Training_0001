import unittest
from src.training.model_utils import calculate_optimal_batch_split


class TestOptimalBatchSplit(unittest.TestCase):
    def test_small_model_4gb_vram(self):
        # 150M model with 4GB VRAM
        per_device, grad_accum = calculate_optimal_batch_split(
            total_batch_size=32,
            vram_gb=4.0,
            n_params=150_000_000,
            seq_len=1024,
        )
        # 150M model (selective checkpointing=True) on 4GB VRAM physically has enough VRAM
        # for per_device=32 without OOM!
        self.assertEqual(per_device, 32)
        self.assertEqual(grad_accum, 1)
        self.assertEqual(per_device * grad_accum, 32)


    def test_small_batch_size(self):
        per_device, grad_accum = calculate_optimal_batch_split(
            total_batch_size=2,
            vram_gb=4.0,
            n_params=150_000_000,
            seq_len=1024,
        )
        self.assertEqual(per_device, 2)
        self.assertEqual(grad_accum, 1)

    def test_prime_or_odd_batch_size(self):
        per_device, grad_accum = calculate_optimal_batch_split(
            total_batch_size=3,
            vram_gb=4.0,
            n_params=150_000_000,
            seq_len=1024,
        )
        self.assertIn(per_device, [1, 3])
        self.assertEqual(per_device * grad_accum, 3)


if __name__ == "__main__":
    unittest.main()
