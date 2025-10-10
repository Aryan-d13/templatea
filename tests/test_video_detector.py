import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import numpy as np

# Add the project root to the Python path
import sys
import os
sys.path.insert(0, 'e:\\Code\\Templatea')

from ..video_detector import detect_and_crop_video

class TestVideoDetector(unittest.TestCase):

    @patch('shutil.copy')
    @patch('subprocess.run')
    @patch('cv2.VideoWriter')
    @patch('cv2.VideoCapture')
    def test_detect_and_crop_video(self, mock_videocapture, mock_videowriter, mock_subprocess_run, mock_shutil_copy):
        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_videocapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [30, 100, 1920, 1080, 0] # fps, total_frames, width, height, other gets

        # Mock frames
        mock_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, mock_frame)

        # Mock VideoWriter
        mock_out = MagicMock()
        mock_videowriter.return_value = mock_out

        # Run the function
        result = detect_and_crop_video('/fake/input.mp4', '/fake/output.mp4', confidence_threshold=10.0)

        # Assertions
        self.assertFalse(result['detected'])
        self.assertEqual(result['output'], '/fake/output.mp4')
        mock_shutil_copy.assert_called_once_with('/fake/input.mp4', '/fake/output.mp4')

if __name__ == '__main__':
    unittest.main()