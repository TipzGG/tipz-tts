import unittest
from unittest.mock import MagicMock

from src.app.dataset import _iter_words, clear_gpu_cache


class DatasetTest(unittest.TestCase):
    def test_iter_words(self):
        word1 = MagicMock()
        word2 = MagicMock()
        s1 = MagicMock()
        s2 = MagicMock()
        s1.words = [word1]
        s2.words = [word2]
        self.assertEqual(_iter_words([s1, s2]), [word1, word2])

    def test_clear_gpu_cache(self):
        torch_module = MagicMock()
        torch_module.cuda.is_available.return_value = True
        clear_gpu_cache(torch_module)
        torch_module.cuda.empty_cache.assert_called_once()


if __name__ == "__main__":
    unittest.main()
