from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from soft_vla.human_intervention.episode_saver import HumanEpisodeSaver


class HumanEpisodeSaverTest(unittest.TestCase):
    def test_collection_compatible_layout_without_depth_and_with_executed_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            saver = HumanEpisodeSaver(tmp, enabled=True, zed_eye="left")
            image = np.zeros((8, 12, 3), dtype=np.uint8)
            image[..., 0] = 255
            saver.record_frame(
                {
                    "timestamp": 0.0,
                    "action_source": "human",
                    "intervention_active": True,
                    "executed_action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0],
                    "u_p12": [0.01 * i for i in range(12)],
                    "u_paw4": [0.0, 3.0, 0.0, 3.0],
                    "state12": list(range(12)),
                    "images": {
                        "observation.images.cam_1": image,
                        "observation.images.cam_2": image,
                        "observation.images.cam_3": image,
                    },
                }
            )
            meta_path = saver.close_episode(success=True, termination_reason="x_success")

            episode_dir = Path(tmp) / "episode_0000"
            self.assertTrue((episode_dir / "images_cam1_zed_left" / "000000.jpg").exists())
            self.assertTrue((episode_dir / "images_cam2" / "000000.jpg").exists())
            self.assertTrue((episode_dir / "images_cam3" / "000000.jpg").exists())
            self.assertFalse((episode_dir / "depth_cam2").exists())
            self.assertFalse((episode_dir / "depth_cam3").exists())

            with (episode_dir / "data.csv").open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["image1_zed_left"], "000000.jpg")
            self.assertEqual(float(rows[0]["u_p1"]), 0.0)
            self.assertAlmostEqual(float(rows[0]["u_p12"]), 0.11)
            self.assertEqual(float(rows[0]["u_paw2"]), 3.0)
            self.assertEqual(float(rows[0]["u_paw4"]), 3.0)
            self.assertEqual(float(rows[0]["executed_action7"]), 1.0)
            self.assertEqual(float(rows[0]["x_ang_radian3"]), 5.0)
            self.assertEqual(float(rows[0]["x_ang_radian_vel3"]), 11.0)

            saved = cv2.imread(str(episode_dir / "images_cam1_zed_left" / "000000.jpg"))
            self.assertIsNotNone(saved)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertTrue(meta["success"])
            self.assertEqual(meta["frame_count"], 1)
            self.assertIn("data_csv", meta)


if __name__ == "__main__":
    unittest.main()
