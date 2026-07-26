import unittest
from src.chinchilla.calculator import determine_optimal_proxy_size


class TestDetermineOptimalProxySize(unittest.TestCase):
    def test_determine_proxy_1_5b(self):
        # 1.5B (1,500,000,000) -> 10% is 150M
        size_str = determine_optimal_proxy_size(1_500_000_000)
        self.assertEqual(size_str, "150M")

    def test_determine_proxy_7b(self):
        # 7B (7,000,000,000) -> 10% is 700M
        size_str = determine_optimal_proxy_size(7_000_000_000)
        self.assertEqual(size_str, "700M")

    def test_determine_proxy_min_bound(self):
        # 100M (100,000,000) -> 10% is 10M, but min bound is 50M
        size_str = determine_optimal_proxy_size(100_000_000)
        self.assertEqual(size_str, "50M")


if __name__ == "__main__":
    unittest.main()
