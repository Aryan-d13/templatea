import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os

# Add the project root to the Python path
import sys
sys.path.insert(0, 'e:\\Code\\Templatea')

from ..gemini_ocr_batch import (
    extract_frame,
    call_gemini,
    generate_ai_one_liners,
    process_video
)

class TestGeminiOcrBatch(unittest.TestCase):

    @patch('cv2.VideoCapture')
    @patch('cv2.imencode')
    def test_extract_frame(self, mock_imencode, mock_videocapture):
        mock_cap = MagicMock()
        mock_videocapture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = [100, 0] # total frames, other gets
        mock_cap.read.return_value = (True, 'fake_frame')
        mock_imencode.return_value = (True, b'fake_buffer')

        frame_bytes = extract_frame(Path('/fake/video.mp4'), 0.5)

        self.assertEqual(frame_bytes, b'fake_buffer')
        mock_videocapture.assert_called_once_with(str(Path('/fake/video.mp4')))
        mock_cap.set.assert_called_once_with(1, 50) # CAP_PROP_POS_FRAMES

    @patch('requests.post')
    def test_call_gemini(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'candidates': [{'content': {'parts': [{'text': 'Hello World'}]}}]
        }
        mock_post.return_value = mock_response

        text = call_gemini('fake_api_key', b'fake_image_bytes', 'fake_prompt')

        self.assertEqual(text, 'Hello World')
        mock_post.assert_called_once()

    @patch('gemini_ocr_batch.call_groq_direct_generation')
    def test_generate_ai_one_liners(self, mock_call_groq):
        mock_call_groq.return_value = ['one', 'two', 'three']

        result = generate_ai_one_liners('ocr_caption', 'fake_perplexity_key', 'fake_groq_key')

        self.assertEqual(len(result['one_liners']), 3)
        self.assertEqual(result['source'], 'groq_direct')
        mock_call_groq.assert_called_once()

    @patch('gemini_ocr_batch.call_gemini')
    @patch('gemini_ocr_batch.extract_frame')
    def test_process_video(self, mock_extract_frame, mock_call_gemini):
        mock_extract_frame.return_value = b'fake_frame_bytes'
        mock_call_gemini.return_value = 'Hello World'

        result = process_video(Path('/fake/video.mp4'), 'fake_api_key', 0.5, 'fake_prompt', 'fake_model')

        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['text'], 'Hello World')
        mock_extract_frame.assert_called_once()
        mock_call_gemini.assert_called_once()

if __name__ == '__main__':
    unittest.main()