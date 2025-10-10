import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import json

# Add the project root to the Python path
import sys
import os
sys.path.insert(0, 'e:\\Code\\Templatea')

from ..template_engine import (
    probe_video_size,
    load_template_cfg,
    measure_wrapped_lines,
    choose_font,
    TemplateRenderRequest,
    TemplateEngine
)

class TestTemplateEngine(unittest.TestCase):

    @patch('subprocess.check_output')
    def test_probe_video_size(self, mock_check_output):
        mock_check_output.return_value = b'1920,1080\n'
        width, height = probe_video_size('/fake/video.mp4')
        self.assertEqual(width, 1920)
        self.assertEqual(height, 1080)
        mock_check_output.assert_called_once()

    def test_load_template_cfg(self):
        template_content = '{"canvas": {"width": 1080, "height": 1920}}'
        with patch('builtins.open', mock_open(read_data=template_content)) as mock_file:
            cfg = load_template_cfg(Path('/fake/template'))
            self.assertEqual(cfg['canvas']['width'], 1080)
            mock_file.assert_called_with(Path('/fake/template/template.json'), 'r', encoding='utf-8')

    def test_measure_wrapped_lines(self):
        mock_font = MagicMock()
        mock_draw = MagicMock()
        mock_draw.textbbox.side_effect = [
            (0, 0, 100, 20),  # "Hello"
            (0, 0, 220, 20),  # "Hello World"
            (0, 0, 110, 20),  # "World"
            (0, 0, 100, 20),  # "Hello"
            (0, 0, 110, 20)   # "World"
        ]

        lines, height = measure_wrapped_lines('Hello World', mock_font, 200, mock_draw)
        self.assertEqual(lines, ['Hello', 'World'])
        self.assertGreater(height, 0)

    @patch('PIL.ImageFont.truetype')
    def test_choose_font(self, mock_truetype):
        mock_font = MagicMock()
        mock_truetype.return_value = mock_font
        mock_draw = MagicMock()
        mock_draw.textbbox.return_value = (0, 0, 100, 20)

        font = choose_font(mock_draw, Path('/fake/font.ttf'), 120, 40, 2, 200, 'Hello World')
        self.assertIsNotNone(font)
        mock_truetype.assert_called()

    @patch('template_engine.composite_canvas_and_video')
    @patch('PIL.Image.new')
    @patch('template_engine.load_template_cfg')
    @patch('template_engine.probe_video_size')
    def test_template_engine_render(self, mock_probe_video_size, mock_load_template_cfg, mock_image_new, mock_composite):
        mock_probe_video_size.return_value = (1080, 1920)
        mock_load_template_cfg.return_value = {
            'canvas': {'width': 1080, 'height': 1920},
            'text': {'font_size': 120, 'max_lines': 2, 'line_width_pct': 90}
        }
        mock_image = MagicMock()
        mock_image_new.return_value = mock_image

        request = TemplateRenderRequest(
            input_video_path='/fake/video.mp4',
            output_video_path='/fake/output.mp4',
            text='Hello World',
            template_root='/fake/template'
        )
        engine = TemplateEngine(request)
        result = engine.render()

        self.assertTrue(result)
        mock_load_template_cfg.assert_called_once()
        mock_probe_video_size.assert_called_once()
        mock_image_new.assert_called_once()
        mock_composite.assert_called_once()

if __name__ == '__main__':
    unittest.main()