from __future__ import annotations

import struct
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from soft_vla.real_robot.pressure_driver import MockPressureDriver, SerialPressureDriver, SerialPressureDriverConfig
from soft_vla.real_robot.safety_manager import SafetyLimits, SafetyManager


class RealRobotSafetyTest(unittest.TestCase):
    def test_pressure_scaling_and_gripper_mapping(self):
        manager = SafetyManager(SafetyLimits(slew_rate_physical_per_s=None))
        cmd = manager.build_pressure_command(
            motion_norm12=np.ones(12, dtype=np.float32),
            gripper_open=1,
            pressure_scale=0.5,
        )
        np.testing.assert_allclose(cmd.motion_physical12, np.full(12, 1.5, dtype=np.float32))
        np.testing.assert_allclose(cmd.gripper_physical4, np.asarray([3, 0, 0, 0], dtype=np.float32))
        self.assertEqual(cmd.final_physical.shape, (16,))

    def test_estop_forces_zero_motion_pressure(self):
        manager = SafetyManager(SafetyLimits(slew_rate_physical_per_s=None))
        manager.request_estop()
        cmd = manager.build_pressure_command(motion_norm12=np.ones(12, dtype=np.float32), gripper_open=0)
        np.testing.assert_allclose(cmd.motion_physical12, np.zeros(12, dtype=np.float32))
        self.assertIn("estop", cmd.safety_flags)

    def test_mock_driver_defaults_to_16_channel_packet(self):
        driver = MockPressureDriver()
        driver.open()
        written = driver.send_physical(np.arange(16, dtype=np.float32))
        self.assertEqual(written, 128)
        np.testing.assert_allclose(driver.packets[-1], np.arange(16, dtype=np.float64))

    def test_mock_driver_can_send_legacy_12_from_16_vector(self):
        driver = MockPressureDriver(packet_channels=12)
        driver.open()
        written = driver.send_physical(np.arange(16, dtype=np.float32))
        self.assertEqual(written, 96)
        np.testing.assert_allclose(driver.packets[-1], np.arange(12, dtype=np.float64))

    def test_serial_driver_defaults_to_16_double_payload(self):
        class FakeSerial:
            instances = []

            def __init__(self, port, baudrate, write_timeout=None):
                self.port = port
                self.baudrate = baudrate
                self.write_timeout = write_timeout
                self.payloads = []
                FakeSerial.instances.append(self)

            def write(self, payload):
                self.payloads.append(payload)
                return len(payload)

            def close(self):
                pass

        fake_serial_module = types.SimpleNamespace(Serial=FakeSerial)
        with patch.dict(sys.modules, {"serial": fake_serial_module}):
            driver = SerialPressureDriver(SerialPressureDriverConfig(port="COM3", baudrate=115200))
            driver.open()
            written = driver.send_physical(np.arange(16, dtype=np.float32))

        payload = FakeSerial.instances[-1].payloads[-1]
        self.assertEqual(written, 128)
        self.assertEqual(len(payload), 128)
        self.assertEqual(struct.unpack("d" * 16, payload), tuple(float(i) for i in range(16)))

    def test_serial_driver_rejects_12_channel_packet(self):
        with self.assertRaisesRegex(ValueError, "packet_channels=16"):
            SerialPressureDriver(SerialPressureDriverConfig(packet_channels=12))


if __name__ == "__main__":
    unittest.main()
