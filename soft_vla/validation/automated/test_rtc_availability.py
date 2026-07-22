from __future__ import annotations

import unittest

from soft_vla.inference.chunk_execution import RTCUnavailableError, probe_official_rtc
from soft_vla.inference.chunk_execution.rtc_executor import RTCExecutor


class RTCAvailabilityTest(unittest.TestCase):
    def test_probe_has_available_key(self):
        result = probe_official_rtc()
        self.assertIn("available", result)

    def test_executor_not_silently_emulated(self):
        with self.assertRaises(RTCUnavailableError):
            RTCExecutor()


if __name__ == "__main__":
    unittest.main()

