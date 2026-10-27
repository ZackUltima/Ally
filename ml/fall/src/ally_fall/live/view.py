"""The on-screen view of the live demo: video, skeleton, fall score, FALL banner (F3c, D-69).

Drawing and key handling only; checked by hand, not by tests (listed as an exclusion in the
sprint review). All logic lives in `pipeline`, `alert` and `app`.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from numpy.typing import NDArray

from ally_fall.live.app import ViewAction
from ally_fall.live.pipeline import FrameResult, LivePipeline

# Skeleton lines between MediaPipe's 33 points (face, arms, trunk, legs, hands, feet).
BONES: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
    (23, 25), (24, 26), (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
    (27, 31), (28, 32), (15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20),
    (16, 22), (18, 20),
)
GREEN = (80, 200, 80)
RED = (40, 40, 230)
WHITE = (240, 240, 240)
VISIBLE = 0.3
CAPTION = "PROTOTYPE: laptop webcam, model decision only (not the phone app, not the INT8 model)"


class CvView:
    def __init__(self, run_label: str, mirror: bool = True, window: str = "Ally fall demo") -> None:
        self._label = run_label
        self._mirror = mirror
        self._window = window
        self._shown_at = time.perf_counter()
        self._fps = 0.0

    def _draw(
        self, bgr: NDArray[np.uint8], result: FrameResult, pipeline: LivePipeline
    ) -> NDArray[np.uint8]:
        height, width = bgr.shape[:2]
        xy = result.xy.copy()
        if self._mirror:
            bgr = cv2.flip(bgr, 1)
            xy[:, 0] = 1.0 - xy[:, 0]
        seen = result.vis >= VISIBLE
        pixels = [(int(x * width), int(y * height)) for x, y in xy]
        colour = RED if pipeline.active else GREEN
        for a, b in BONES:
            if seen[a] and seen[b]:
                cv2.line(bgr, pixels[a], pixels[b], colour, 2, cv2.LINE_AA)
        for i in np.flatnonzero(seen):
            cv2.circle(bgr, pixels[int(i)], 3, WHITE, -1, cv2.LINE_AA)
        score = pipeline.last_score
        rule = pipeline.rule
        lines = [
            f"model {self._label}",
            "warming up" if score is None else f"fall score {score:.2f}",
            f"alert when score >= {rule.threshold:.2f} on {rule.required} ticks in a row",
            f"alerts so far {pipeline.fired_total}   {self._fps:.0f} fps",
        ]
        if not seen.any():
            lines.append("no person found")
        for row, text in enumerate(lines):
            # A dark box behind the text (OpenCV 5 spaces a thick outline differently).
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            top = 22 + 22 * row - text_h - 4
            cv2.rectangle(bgr, (6, top), (14 + text_w, top + text_h + 10), (30, 30, 30), -1)
            cv2.putText(bgr, text, (10, 22 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        WHITE, 1, cv2.LINE_AA)
        bar_top = height - 40
        cv2.rectangle(bgr, (10, bar_top), (width - 10, bar_top + 14), (60, 60, 60), -1)
        fill = 0 if score is None else int((width - 20) * score)
        cv2.rectangle(bgr, (10, bar_top), (10 + fill, bar_top + 14), colour, -1)
        mark = 10 + int((width - 20) * rule.threshold)
        cv2.line(bgr, (mark, bar_top - 4), (mark, bar_top + 18), WHITE, 2)
        cv2.putText(bgr, CAPTION, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    WHITE, 1, cv2.LINE_AA)
        if pipeline.active:
            cv2.rectangle(bgr, (0, 0), (width - 1, height - 1), RED, 8)
            (text_w, text_h), _ = cv2.getTextSize("FALL", cv2.FONT_HERSHEY_SIMPLEX, 2.2, 4)
            left, top = width - text_w - 30, 14
            cv2.rectangle(bgr, (left - 10, top), (width - 14, top + text_h + 28), (30, 30, 30), -1)
            cv2.putText(bgr, "FALL", (left, top + text_h + 12), cv2.FONT_HERSHEY_SIMPLEX, 2.2,
                        RED, 4, cv2.LINE_AA)
        return bgr

    def show(
        self, rgb: NDArray[np.uint8], result: FrameResult, pipeline: LivePipeline
    ) -> ViewAction:
        now = time.perf_counter()
        self._fps = 0.9 * self._fps + 0.1 / max(now - self._shown_at, 1e-6)
        self._shown_at = now
        bgr = self._draw(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), result, pipeline)
        cv2.imshow(self._window, bgr)
        action = self._key(cv2.waitKey(1) & 0xFF, pipeline)
        if cv2.getWindowProperty(self._window, cv2.WND_PROP_VISIBLE) < 1:
            return ViewAction.QUIT
        while action is None:  # paused: wait for space or q
            action = self._key(cv2.waitKey(50) & 0xFF, pipeline, paused=True)
        return action

    def _key(self, key: int, pipeline: LivePipeline, paused: bool = False) -> ViewAction | None:
        rule = pipeline.rule
        if key in (ord("q"), 27):
            return ViewAction.QUIT
        if key == ord(" "):
            return ViewAction.NONE if paused else None
        if key == ord("r"):
            return ViewAction.RESET
        if key == ord("s"):
            return ViewAction.SAVE_KEYPOINTS
        if key == ord("f"):
            return ViewAction.SAVE_FRAME
        if key == ord("m"):
            self._mirror = not self._mirror
        elif key in (ord("+"), ord("=")):
            rule.set_threshold(min(1.0, round(rule.threshold + 0.05, 2)))
        elif key == ord("-"):
            rule.set_threshold(max(0.05, round(rule.threshold - 0.05, 2)))
        elif key == ord("]"):
            rule.set_required(rule.required + 1)
        elif key == ord("["):
            rule.set_required(max(1, rule.required - 1))
        return ViewAction.NONE if not paused else None

