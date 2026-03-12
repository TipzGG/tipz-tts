import unittest

from src.app.speaker_isolation import _choose_speaker_by_duration, _has_overlap


class SpeakerIsolationHelpersTest(unittest.TestCase):
    def test_has_overlap(self):
        self.assertTrue(_has_overlap((1.0, 2.0), [(0.5, 1.5)]))
        self.assertFalse(_has_overlap((2.0, 3.0), [(0.0, 2.0)]))

    def test_choose_speaker_by_duration(self):
        selected = _choose_speaker_by_duration(
            {
                "SPEAKER_00": [(0.0, 1.0), (2.0, 3.0)],
                "SPEAKER_01": [(0.0, 4.0)],
            }
        )
        self.assertEqual(selected, "SPEAKER_01")


if __name__ == "__main__":
    unittest.main()
