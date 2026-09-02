import io
import unittest
from unittest.mock import patch

from crossword import progress


class TestProgressLog(unittest.TestCase):
    def tearDown(self):
        progress.disable()

    def test_silent_when_disabled(self):
        buf = io.StringIO()
        with patch("crossword.progress.sys.stderr", buf):
            progress.log("hello")
        self.assertEqual(buf.getvalue(), "")

    def test_prints_when_enabled(self):
        buf = io.StringIO()
        progress.enable()
        with patch("crossword.progress.sys.stderr", buf):
            progress.log("hello")
        self.assertIn("hello", buf.getvalue())
