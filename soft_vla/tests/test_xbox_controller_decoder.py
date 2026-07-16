from __future__ import annotations

import unittest
from types import SimpleNamespace

import evdev

from soft_vla.human_intervention.xbox_controller import EvdevXboxStateDecoder, _button_from_evdev_name


class XboxControllerDecoderTest(unittest.TestCase):
    def test_button_aliases_accept_single_names(self):
        self.assertEqual(_button_from_evdev_name("BTN_SOUTH"), "a")
        self.assertEqual(_button_from_evdev_name("BTN_EAST"), "b")
        self.assertEqual(_button_from_evdev_name("BTN_WEST"), "x")
        self.assertEqual(_button_from_evdev_name("BTN_NORTH"), "y")

    def test_button_aliases_accept_evdev_alias_lists(self):
        self.assertEqual(_button_from_evdev_name(["BTN_A", "BTN_GAMEPAD", "BTN_SOUTH"]), "a")
        self.assertEqual(_button_from_evdev_name(["BTN_B", "BTN_EAST"]), "b")
        self.assertEqual(_button_from_evdev_name(["BTN_X", "BTN_WEST"]), "x")
        self.assertEqual(_button_from_evdev_name(["BTN_Y", "BTN_NORTH"]), "y")

    def test_decoder_uses_ev_key_bytype_names(self):
        decoder = EvdevXboxStateDecoder(SimpleNamespace())
        code_to_button = {
            evdev.ecodes.BTN_A: "a",
            evdev.ecodes.BTN_B: "b",
            evdev.ecodes.BTN_X: "x",
            evdev.ecodes.BTN_Y: "y",
        }
        for code, button in code_to_button.items():
            state = decoder.update(SimpleNamespace(type=evdev.ecodes.EV_KEY, code=code, value=1))
            self.assertTrue(state["buttons"][button])
            state = decoder.update(SimpleNamespace(type=evdev.ecodes.EV_KEY, code=code, value=0))
            self.assertFalse(state["buttons"][button])


if __name__ == "__main__":
    unittest.main()
