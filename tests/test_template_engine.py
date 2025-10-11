import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

import template_engine as engine


def test_load_template_cfg_reads_json(tmp_path):
    template_dir = tmp_path / "template"
    template_dir.mkdir()
    expected = {"canvas": {"width": 800, "height": 600}}
    (template_dir / "template.json").write_text(json.dumps(expected), encoding="utf-8")

    cfg = engine.load_template_cfg(template_dir)

    assert cfg == expected


def test_load_template_cfg_missing_file(tmp_path):
    template_dir = tmp_path / "template"
    template_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        engine.load_template_cfg(template_dir)


def test_probe_video_size_success(monkeypatch):
    called = {}

    def fake_check_output(cmd, stderr=None):
        called["cmd"] = cmd
        return b"640,480\n"

    monkeypatch.setattr(engine.subprocess, "check_output", fake_check_output)

    width, height = engine.probe_video_size("video.mp4")

    assert (width, height) == (640, 480)
    assert called["cmd"][0] == "ffprobe"


def test_probe_video_size_returns_default_on_error(monkeypatch):
    def fake_check_output(cmd, stderr=None):
        raise engine.subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(engine.subprocess, "check_output", fake_check_output)

    assert engine.probe_video_size("missing.mp4") == (1080, 1920)


def test_probe_duration_success(monkeypatch):
    monkeypatch.setattr(engine.subprocess, "check_output", lambda *_, **__: b"3.5\n")

    assert engine.probe_duration("video.mp4") == pytest.approx(3.5)


def test_probe_duration_handles_error(monkeypatch):
    monkeypatch.setattr(engine.subprocess, "check_output", lambda *_, **__: (_ for _ in ()).throw(Exception("ffprobe failed")))

    assert engine.probe_duration("broken.mp4") is None


def test_measure_wrapped_lines_wraps_and_counts_height():
    image = Image.new("RGB", (200, 200), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    text = "This is a moderately long sentence that should wrap more than once."
    lines, total_height = engine.measure_wrapped_lines(text, font, max_width=70, draw=draw)

    assert len(lines) >= 2
    assert total_height > 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        assert bbox[2] <= 75  # small cushion for rounding


def test_choose_font_falls_back_to_default(monkeypatch):
    call_count = {"value": 0}

    original_truetype = engine.ImageFont.truetype

    def fake_truetype(path, size, *args, **kwargs):
        if "layout_engine" in kwargs:
            return original_truetype(path, size, *args, **kwargs)
        call_count["value"] += 1
        raise OSError("missing font")

    monkeypatch.setattr(engine.ImageFont, "truetype", fake_truetype)

    image = Image.new("RGB", (200, 200))
    draw = ImageDraw.Draw(image)

    font = engine.choose_font(
        draw,
        Path("missing.ttf"),
        requested_size=42,
        min_size=12,
        max_lines=2,
        max_width=120,
        text="Fallback font selection test",
    )

    assert isinstance(font, ImageFont.FreeTypeFont)
    assert call_count["value"] >= 1


def test_apply_text_casing_modes():
    assert engine.apply_text_casing("hello world", "upper") == "HELLO WORLD"
    assert engine.apply_text_casing("HELLO WORLD", "lowercase") == "hello world"
    assert engine.apply_text_casing("hELLo world", "sentence") == "Hello world"
    assert engine.apply_text_casing("hello world", "title") == "Hello World"
    assert engine.apply_text_casing("Keep Original", None) == "Keep Original"


def test_parse_highlight_words_supports_phrases():
    manual = ["Samsung", "AI powered home helper", 123]

    result = engine.parse_highlight_words(manual)

    assert result == [["Samsung"], ["AI", "powered", "home", "helper"], ["123"]]


def test_find_phrase_positions_handles_multi_line_phrase():
    lines = ["AI powered", "home helper robot"]
    phrase = ["powered", "home", "helper"]

    positions = engine.find_phrase_positions(lines, phrase)

    assert positions == [(0, 1, 1), (1, 0, 0), (1, 1, 1)]


def test_select_highlight_words_via_ai_prefers_longer_tokens():
    text = "AI powered highlight demonstration context"

    selected = engine.select_highlight_words_via_ai(text, top_k=2)

    assert selected == [["demonstration"], ["highlight"]]


def test_template_engine_render_renders_canvas(tmp_path, monkeypatch):
    template_root = tmp_path / "template"
    template_root.mkdir()
    config = {
        "canvas": {"width": 400, "height": 400},
        "background": {"type": "color", "value": "#000000"},
        "video": {"width_pct": 50, "vertical_align": "center"},
        "text": {
            "font": "",
            "font_size": 48,
            "min_font_size": 24,
            "max_lines": 3,
            "line_width_pct": 80,
            "align": "center",
            "position_relative_to_video": {"gap_px": 20},
            "color": "#ffffff",
            "highlight": {"enabled": True, "mode": "manual", "manual_words": ["Hello"]},
        },
        "logo": {"enabled": False},
    }
    (template_root / "template.json").write_text(json.dumps(config), encoding="utf-8")

    input_video_path = tmp_path / "input.mp4"
    input_video_path.write_bytes(b"fake")
    output_video_path = tmp_path / "output.mp4"

    def fake_probe_video_size(path):
        assert path == str(input_video_path)
        return 1280, 720

    captured = {}

    def fake_composite(canvas_path, in_path, out_path, cfg):
        assert Path(canvas_path).exists()
        assert in_path == str(input_video_path)
        assert out_path == str(output_video_path)
        captured["canvas_path"] = Path(canvas_path)
        captured["cfg"] = cfg
        Path(out_path).write_text("rendered")

    monkeypatch.setattr(engine, "probe_video_size", fake_probe_video_size)
    monkeypatch.setattr(engine, "composite_canvas_and_video", fake_composite)

    request = engine.TemplateRenderRequest(
        input_video_path=str(input_video_path),
        output_video_path=str(output_video_path),
        text="Hello world from template engine",
        template_root=str(template_root),
    )
    result = engine.TemplateEngine(request).render()

    assert result is True
    assert "canvas_path" in captured
    assert not captured["canvas_path"].exists()
    assert output_video_path.read_text() == "rendered"
    assert captured["cfg"]["canvas"]["width"] == 400


def test_template_engine_applies_casing_before_layout(tmp_path, monkeypatch):
    template_root = tmp_path / "template"
    template_root.mkdir()
    config = {
        "canvas": {"width": 300, "height": 300},
        "background": {"type": "color", "value": "#000000"},
        "video": {"width_pct": 50, "vertical_align": "center"},
        "text": {
            "font": "",
            "font_size": 32,
            "min_font_size": 16,
            "max_lines": 2,
            "line_width_pct": 80,
            "align": "center",
            "position_relative_to_video": {"gap_px": 20},
            "color": "#ffffff",
            "highlight": {"enabled": False},
            "casing_mode": "upper",
        },
        "logo": {"enabled": False},
    }
    (template_root / "template.json").write_text(json.dumps(config), encoding="utf-8")

    input_video_path = tmp_path / "input.mp4"
    input_video_path.write_bytes(b"fake")
    output_video_path = tmp_path / "output.mp4"

    monkeypatch.setattr(engine, "probe_video_size", lambda _: (640, 360))

    measured = {}

    def fake_measure(text, font, max_width, draw, **kwargs):
        measured["text"] = text
        return [text], 40

    monkeypatch.setattr(engine, "measure_wrapped_lines", fake_measure)

    def fake_composite(canvas_path, *_args, **_kwargs):
        Path(canvas_path).unlink(missing_ok=True)
        output_video_path.write_text("rendered")
        return True

    monkeypatch.setattr(engine, "composite_canvas_and_video", fake_composite)

    request = engine.TemplateRenderRequest(
        input_video_path=str(input_video_path),
        output_video_path=str(output_video_path),
        text="make me loud",
        template_root=str(template_root),
    )

    assert engine.TemplateEngine(request).render() is True
    assert measured["text"] == "MAKE ME LOUD"
    assert output_video_path.read_text() == "rendered"


def test_template_engine_supports_root_level_casing_key(tmp_path, monkeypatch):
    template_root = tmp_path / "template"
    template_root.mkdir()
    config = {
        "canvas": {"width": 300, "height": 300},
        "background": {"type": "color", "value": "#000000"},
        "video": {"width_pct": 50, "vertical_align": "center"},
        "text": {
            "font": "",
            "font_size": 32,
            "min_font_size": 16,
            "max_lines": 2,
            "line_width_pct": 80,
            "align": "center",
            "position_relative_to_video": {"gap_px": 20},
            "color": "#ffffff",
            "highlight": {"enabled": False},
        },
        "text.casing_mode": "upper",
        "logo": {"enabled": False},
    }
    (template_root / "template.json").write_text(json.dumps(config), encoding="utf-8")

    input_video_path = tmp_path / "input.mp4"
    input_video_path.write_bytes(b"fake")
    output_video_path = tmp_path / "output.mp4"

    monkeypatch.setattr(engine, "probe_video_size", lambda _: (640, 360))

    measured = {}

    def fake_measure(text, font, max_width, draw, **kwargs):
        measured["text"] = text
        return [text], 40

    monkeypatch.setattr(engine, "measure_wrapped_lines", fake_measure)

    def fake_composite(canvas_path, *_args, **_kwargs):
        Path(canvas_path).unlink(missing_ok=True)
        output_video_path.write_text("rendered")
        return True

    monkeypatch.setattr(engine, "composite_canvas_and_video", fake_composite)

    request = engine.TemplateRenderRequest(
        input_video_path=str(input_video_path),
        output_video_path=str(output_video_path),
        text="make me loud",
        template_root=str(template_root),
    )

    assert engine.TemplateEngine(request).render() is True
    assert measured["text"] == "MAKE ME LOUD"
    assert output_video_path.read_text() == "rendered"


def test_template_engine_reads_root_level_line_spacing(tmp_path, monkeypatch):
    template_root = tmp_path / "template"
    template_root.mkdir()
    config = {
        "canvas": {"width": 300, "height": 300},
        "background": {"type": "color", "value": "#000000"},
        "video": {"width_pct": 50, "vertical_align": "center"},
        "text": {
            "font": "",
            "font_size": 32,
            "min_font_size": 16,
            "max_lines": 2,
            "line_width_pct": 80,
            "align": "center",
            "position_relative_to_video": {"gap_px": 20},
            "color": "#ffffff",
            "highlight": {"enabled": False},
        },
        "text.line_spacing_factor": 1.1,
    }
    (template_root / "template.json").write_text(json.dumps(config), encoding="utf-8")

    input_video_path = tmp_path / "input.mp4"
    input_video_path.write_bytes(b"fake")
    output_video_path = tmp_path / "output.mp4"

    monkeypatch.setattr(engine, "probe_video_size", lambda _: (640, 360))

    captured_factor = {}

    def fake_measure(text, font, max_width, draw, **kwargs):
        captured_factor["value"] = kwargs.get("line_spacing_factor")
        return [text], 40

    monkeypatch.setattr(engine, "measure_wrapped_lines", fake_measure)
    def fake_composite(canvas_path, *_args, **_kwargs):
        Path(canvas_path).unlink(missing_ok=True)
        output_video_path.write_text("rendered")
        return True

    monkeypatch.setattr(engine, "composite_canvas_and_video", fake_composite)

    request = engine.TemplateRenderRequest(
        input_video_path=str(input_video_path),
        output_video_path=str(output_video_path),
        text="spacing test",
        template_root=str(template_root),
    )

    assert engine.TemplateEngine(request).render() is True
    assert captured_factor["value"] == 1.1
