"""Tests for the Stage 4 caption style: emphasis colours, the
word sweep at render, and the phrase pop.

Emphasis is a pure function of the cue text (numbers + named
entities in Viral Pink #FF3D7E); the sweep exists only where real
word timings drive it; with no colours in play the caption layer
is the historic single-draw render, byte for byte.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_director.caption_align import CaptionCue  # noqa: E402
from spatial_director.compositor import read_video_frames  # noqa: E402
from spatial_director.layout_law import (  # noqa: E402
    CAPTION_EMPHASIS_RGBA,
    caption_emphasis_indices,
    caption_layer,
    caption_word_colors,
    initial_caption_rect,
)
from spatial_director.rescue_loop import (  # noqa: E402
    _pop_caption_layer,
    render_composite_with_captions,
)
from spatial_director.word_timing import WordTiming  # noqa: E402

CANVAS = (1080, 1920)
RECT = initial_caption_rect()
PINK = np.array(CAPTION_EMPHASIS_RGBA[:3], dtype=int)


def pink_px(layer: np.ndarray) -> int:
    rgb = layer[..., :3].astype(int)
    near = np.abs(rgb - PINK).sum(axis=2) < 30
    return int((near & (layer[..., 3] > 0)).sum())


class EmphasisIndexTests(unittest.TestCase):
    def test_digit_tokens_are_emphasis(self):
        idx = caption_emphasis_indices("He ran 28,000 kilometres home")
        self.assertIn(2, idx)
        self.assertNotIn(0, idx)
        self.assertNotIn(1, idx)

    def test_number_words_are_emphasis(self):
        idx = caption_emphasis_indices("twenty people saw it")
        self.assertIn(0, idx)

    def test_named_entities_mid_sentence(self):
        idx = caption_emphasis_indices("the NASA crew thanked Megan")
        self.assertIn(1, idx)   # NASA (acronym)
        self.assertIn(4, idx)   # Megan (proper noun)
        self.assertNotIn(0, idx)

    def test_sentence_start_and_pronoun_i_excluded(self):
        # A capital that opens a sentence is not flagged by
        # position alone: "Maria" after a full stop is a proper
        # noun the heuristic conservatively misses (documented);
        # digits in the same position still land.
        idx = caption_emphasis_indices("The crew left. Maria stayed")
        self.assertNotIn(0, idx)
        self.assertNotIn(3, idx)
        idx2 = caption_emphasis_indices("then I saw 4 ships")
        self.assertNotIn(1, idx2)  # the pronoun I
        self.assertIn(3, idx2)     # the digit

    def test_plain_text_colours_nothing(self):
        self.assertIsNone(caption_word_colors("the band plays on"))
        self.assertIsNotNone(
            caption_word_colors("the band plays 4 songs"))


class CaptionLayerColourTests(unittest.TestCase):
    def test_emphasis_word_renders_pink_plain_does_not(self):
        text = "He ran 42 miles"
        colored = caption_layer(
            text, RECT, CANVAS,
            word_colors=caption_word_colors(text))
        self.assertGreater(pink_px(colored), 50)
        plain = caption_layer(text, RECT, CANVAS)
        self.assertEqual(pink_px(plain), 0)

    def test_no_colors_is_byte_identical_to_empty_colors(self):
        a = caption_layer("The band never rehearses.", RECT, CANVAS)
        b = caption_layer("The band never rehearses.", RECT, CANVAS,
                          word_colors={})
        self.assertTrue(np.array_equal(a, b))

    def test_sweep_index_colours_that_word(self):
        text = "alpha beta gamma"
        layer = caption_layer(
            text, RECT, CANVAS,
            word_colors=caption_word_colors(text, active_index=1))
        self.assertGreater(pink_px(layer), 50)
        plain = caption_layer(text, RECT, CANVAS)
        self.assertEqual(pink_px(plain), 0)


class PopTests(unittest.TestCase):
    def test_pop_scale_one_is_identity(self):
        layer = caption_layer("He ran 42 miles", RECT, CANVAS)
        self.assertTrue(np.array_equal(
            _pop_caption_layer(layer, RECT, 1.0), layer))

    def test_pop_scales_the_block_only(self):
        layer = caption_layer("He ran 42 miles", RECT, CANVAS)
        popped = _pop_caption_layer(layer, RECT, 1.05)
        # Alpha mass is (nearly) preserved - the pop rescales,
        # it does not add or remove ink wholesale.
        mass0 = int((layer[..., 3] > 0).sum())
        mass1 = int((popped[..., 3] > 0).sum())
        self.assertLess(abs(mass1 - mass0) / mass0, 0.2)
        # Far from the block, nothing moved.
        self.assertTrue(np.array_equal(
            popped[:200, :, :], layer[:200, :, :]))


def _tiny_render(out: Path, *, timings=None, pop_s=0.0):
    frames = [np.full((1920, 1080, 3), 40, dtype=np.uint8)
              for _ in range(9)]
    alphas = [np.zeros((1920, 1080), dtype=np.float32)
              for _ in range(9)]
    cues = [CaptionCue("alpha beta gamma", 0.0, 3.0)]
    kwargs = {}
    if timings is not None:
        kwargs["caption_word_timings"] = timings
    if pop_s:
        kwargs["caption_pop_s"] = pop_s
    render_composite_with_captions(
        frames, alphas, hook_rgba=None, hook_window=None,
        cues=cues, caption_rect=RECT, audio_from=None,
        out_path=out, fps=3.0, canvas=CANVAS, **kwargs)
    return read_video_frames(out)


def _frame_pink(frame: np.ndarray) -> int:
    rgb = frame[..., :3].astype(int)
    return int((np.abs(rgb - PINK).sum(axis=2) < 60).sum())


class RenderSweepTests(unittest.TestCase):
    def test_sweep_highlight_appears_only_with_timings(self):
        timings = [[
            WordTiming("alpha", 0.0, 1.0, "dsp"),
            WordTiming("beta", 1.0, 2.0, "dsp"),
            WordTiming("gamma", 2.0, 3.0, "dsp"),
        ]]
        with tempfile.TemporaryDirectory() as td:
            swept = _tiny_render(Path(td) / "swept.mp4",
                                 timings=timings)
            plain = _tiny_render(Path(td) / "plain.mp4")
        # Frame at ~1.5s (index 4-5): "beta" is the active word.
        self.assertGreater(_frame_pink(swept[4]), 30)
        self.assertEqual(_frame_pink(plain[4]), 0)


if __name__ == "__main__":
    unittest.main()
