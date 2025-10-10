import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import os
import json

# Add the project root to the Python path
import sys
sys.path.insert(0, 'e:\\Code\\Templatea')

from ..orchestrator import (
    derive_canonical_from_url,
    clean_text_for_render,
    normalize_dashes,
    write_status,
    read_meta,
    write_meta,
    process_single_workspace
)

class TestOrchestrator(unittest.TestCase):

    def test_derive_canonical_from_url(self):
        self.assertEqual(derive_canonical_from_url('https://www.instagram.com/reel/Cxyz...'), 'Cxyz...')
        self.assertEqual(derive_canonical_from_url('https://www.instagram.com/p/Cxyz...'), 'Cxyz...')
        self.assertEqual(derive_canonical_from_url('https://www.instagram.com/tv/Cxyz...'), 'Cxyz...')
        self.assertEqual(derive_canonical_from_url('https://www.instagram.com/reels/Cxyz...'), 'Cxyz...')
        self.assertEqual(derive_canonical_from_url('https://www.instagram.com/video/Cxyz...'), 'Cxyz...')
        self.assertIsNone(derive_canonical_from_url('https://www.instagram.com/'))

    def test_clean_text_for_render(self):
        self.assertEqual(clean_text_for_render('Hello 😀 World'), 'Hello  World')
        self.assertEqual(clean_text_for_render('This is a test.'), 'This is a test.')
        self.assertEqual(clean_text_for_render(''), '')

    def test_normalize_dashes(self):
        self.assertEqual(normalize_dashes('This is a test\u2014with a dash.'), 'This is a test, with a dash.')
        self.assertEqual(normalize_dashes('This is a test\u2013with another dash.'), 'This is a test, with another dash.')
        self.assertEqual(normalize_dashes('No dashes here.'), 'No dashes here.')

    @patch('os.replace')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_write_status(self, mock_open, mock_replace):
        ws = Path('/fake/workspace')
        write_status(ws, 'test_step', 'success', error='test_error', retries=1)
        mock_open.assert_called_with(ws / 'test_step.status.tmp', 'w', encoding='utf8')
        handle = mock_open()
        written_data = handle.write.call_args[0][0]
        self.assertIn('"status": "success"', written_data)
        self.assertIn('"error": "test_error"', written_data)
        self.assertIn('"retries": 1', written_data)
        mock_replace.assert_called_once_with(str(ws / 'test_step.status.tmp'), str(ws / 'test_step.status'))

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"key": "value"}')
    def test_read_meta(self, mock_open):
        ws = Path('/fake/workspace')
        (ws / 'meta.json').touch()
        meta = read_meta(ws)
        self.assertEqual(meta, {'key': 'value'})
        mock_open.assert_called_with(ws / 'meta.json', 'r', encoding='utf-8-sig')

    @patch('os.replace')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_write_meta(self, mock_open, mock_replace):
        ws = Path('/fake/workspace')
        write_meta(ws, {'key': 'value'})
        mock_open.assert_called_with(ws / 'meta.json.tmp', 'w', encoding='utf8')
        handle = mock_open()
        written_data = handle.write.call_args[0][0]
        self.assertIn('"key": "value"', written_data)
        mock_replace.assert_called_once_with(str(ws / 'meta.json.tmp'), str(ws / 'meta.json'))

    @patch('orchestrator.release_lock')
    @patch('orchestrator.acquire_lock')
    @patch('orchestrator.ensure_render')
    @patch('orchestrator.ensure_choice')
    @patch('orchestrator.ensure_ocr')
    @patch('orchestrator.ensure_detector')
    @patch('orchestrator.write_meta')
    @patch('orchestrator.read_meta')
    def test_process_single_workspace(self, mock_read_meta, mock_write_meta, mock_ensure_detector, mock_ensure_ocr, mock_ensure_choice, mock_ensure_render, mock_acquire_lock, mock_release_lock):
        ws = Path('/fake/workspace')
        mock_acquire_lock.return_value = True
        mock_read_meta.return_value = {}
        mock_ensure_detector.return_value = True
        mock_ensure_ocr.return_value = True
        mock_ensure_choice.return_value = True
        mock_ensure_render.return_value = True

        summary = process_single_workspace(ws, auto=True, template_id='test_template')

        self.assertEqual(summary['id'], 'workspace')
        self.assertTrue(summary['detected'])
        self.assertTrue(summary['ocr'])
        self.assertTrue(summary['choice'])
        self.assertTrue(summary['rendered'])

        mock_acquire_lock.assert_called_once_with(ws)
        mock_release_lock.assert_called_once_with(ws)
        mock_ensure_detector.assert_called_once()
        mock_ensure_ocr.assert_called_once()
        mock_ensure_choice.assert_called_once()
        mock_ensure_render.assert_called_once()

if __name__ == '__main__':
    unittest.main()