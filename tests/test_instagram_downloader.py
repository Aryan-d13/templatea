import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add the project root to the Python path
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..instagram_downloader import (
    extract_shortcode,
    download_instagram_reel_efficient
)

class TestInstagramDownloader(unittest.TestCase):

    def test_extract_shortcode(self):
        self.assertEqual(extract_shortcode('https://www.instagram.com/reel/Cxyz...'), 'Cxyz...')
        self.assertEqual(extract_shortcode('https://www.instagram.com/p/Cxyz...'), 'Cxyz...')
        self.assertEqual(extract_shortcode('https://www.instagram.com/tv/Cxyz...'), 'Cxyz...')
        self.assertIsNone(extract_shortcode('https://www.instagram.com/'))

    @patch('instagram_downloader.try_ytdlp_to_dir')
    @patch('instagram_downloader.try_instaloader_to_dir')
    @patch('instagram_downloader.normalize_and_write_meta')
    @patch('instagram_downloader.ensure_workspace')
    def test_download_instagram_reel_efficient_instaloader_success(self, mock_ensure_workspace, mock_normalize_and_write_meta, mock_try_instaloader, mock_try_ytdlp):
        mock_ensure_workspace.return_value = (Path('/fake/workspace/Cxyz'), Path('/fake/workspace/Cxyz/00_raw'))
        mock_try_instaloader.return_value = True
        mock_normalize_and_write_meta.return_value = Path('/fake/workspace/Cxyz')

        ws = download_instagram_reel_efficient('https://www.instagram.com/reel/Cxyz...')

        self.assertIsNotNone(ws)
        self.assertEqual(ws, Path('/fake/workspace/Cxyz'))
        mock_try_instaloader.assert_called_once()
        mock_try_ytdlp.assert_not_called()

    @patch('instagram_downloader.try_ytdlp_to_dir')
    @patch('instagram_downloader.try_instaloader_to_dir')
    @patch('instagram_downloader.normalize_and_write_meta')
    @patch('instagram_downloader.ensure_workspace')
    def test_download_instagram_reel_efficient_ytdlp_fallback(self, mock_ensure_workspace, mock_normalize_and_write_meta, mock_try_instaloader, mock_try_ytdlp):
        mock_ensure_workspace.return_value = (Path('/fake/workspace/Cxyz'), Path('/fake/workspace/Cxyz/00_raw'))
        mock_try_instaloader.return_value = False
        mock_try_ytdlp.return_value = True
        mock_normalize_and_write_meta.return_value = Path('/fake/workspace/Cxyz')

        ws = download_instagram_reel_efficient('https://www.instagram.com/reel/Cxyz...')

        self.assertIsNotNone(ws)
        self.assertEqual(ws, Path('/fake/workspace/Cxyz'))
        mock_try_instaloader.assert_called_once()
        mock_try_ytdlp.assert_called_once()

    @patch('instagram_downloader.try_ytdlp_to_dir')
    @patch('instagram_downloader.try_instaloader_to_dir')
    def test_download_instagram_reel_efficient_all_fail(self, mock_try_instaloader, mock_try_ytdlp):
        mock_try_instaloader.return_value = False
        mock_try_ytdlp.return_value = False

        ws = download_instagram_reel_efficient('https://www.instagram.com/reel/Cxyz...')

        self.assertIsNone(ws)
        mock_try_instaloader.assert_called_once()
        mock_try_ytdlp.assert_called_once()

if __name__ == '__main__':
    unittest.main()