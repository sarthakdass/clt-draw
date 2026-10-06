"""The Pygame interface: draw a density, lock it in, then step n = 2, 4, ..., 1024."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pygame

from .core import LEVELS, Level, Z_GRID, Z_MAX, iter_levels, normal_pdf, values_from_columns

TITLE = "Central Limit Theorem Visualization"
DESCRIPTION = ("Behold the power of the Central Limit Theorem! Convergence to a normal distribution is "
               "inevitable, how long can your curve avoid it?")
CREDIT = ("The idea for this project is not original. An anonymous student from Professor Dan Ostrov's "
          "Probability class designed a similar website, and I took inspiration in designing my own "
          "visually interactive Central Limit Theorem project.")

# palette shared with the website
INK, PANEL, RULE = (13, 16, 22), (21, 25, 34), (38, 45, 60)
TEXT, SOFT = (233, 236, 242), (163, 172, 189)
ACCENT, LILAC = (255, 90, 54), (143, 155, 255)

WIDTH, HEIGHT = 1100, 800
PLOT = pygame.Rect(40, 122, 1020, 380)
RES = 1000                                        # drawing columns (resolution-independent)
N_VALUES = [2 ** i for i in range(1, LEVELS + 1)]


def _wrap(font: pygame.font.Font, text: str, width: int) -> List[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if font.size(trial)[0] <= width:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        names = "inter,ibmplexsans,dejavusans,arial,helvetica"
        self.f_title = pygame.font.SysFont(names, 32, bold=True)
        self.f_body = pygame.font.SysFont(names, 17)
        self.f_small = pygame.font.SysFont(names, 14)
        self.f_mono = pygame.font.SysFont("ibmplexmono,dejavusansmono,couriernew,monospace", 15)
        self.clock = pygame.time.Clock()

        self.x0, self.x1 = PLOT.left + 24, PLOT.right - 24
        self.top, self.base = PLOT.top + 48, PLOT.bottom - 38

        self.submit_rect = pygame.Rect(40, 520, 170, 44)
        self.redraw_rect = pygame.Rect(224, 520, 130, 44)
        self.normal_rect = pygame.Rect(400, 531, 22, 22)
        self.orig_rect = pygame.Rect(560, 531, 22, 22)
        self.rail_xs = np.linspace(86, 600, LEVELS)
        self.rail_y = 650
        self.chart_rect = pygame.Rect(676, 580, 384, 156)

        self.show_normal, self.show_original = True, True
        self.reset()

    # ---------------------------------------------------------------- state
    def reset(self) -> None:
        self.cols = np.full(RES, np.nan)
        self.last_col: Optional[int] = None
        self.last_val = 0.0
        self.drawing = False
        self.locked = False
        self.levels: List[Level] = []
        self._gen = None
        self.sel = 0                           # index into N_VALUES
        self.shown: Optional[np.ndarray] = None
        self.shown_ymax = 0.5
        self.message = ""

    @property
    def ready(self) -> bool:
        return self.locked and self._gen is None and len(self.levels) == LEVELS + 1

    def can_submit(self) -> bool:
        return not self.locked and values_from_columns(self.cols) is not None

    # --------------------------------------------------------------- drawing
    def _to_col_val(self, pos):
        c = (pos[0] - self.x0) / (self.x1 - self.x0) * (RES - 1)
        v = (self.base - pos[1]) / (self.base - self.top)
        return int(round(min(max(c, 0), RES - 1))), v

    def _paint(self, pos) -> None:
        col, val = self._to_col_val(pos)
        if self.last_col is None:
            self.cols[col] = val
        else:                                   # fill every column the stroke crossed
            a, b = sorted((self.last_col, col))
            for c in range(a, b + 1):
                t = 0.0 if b == a else (c - self.last_col) / (col - self.last_col)
                self.cols[c] = self.last_val + t * (val - self.last_val)
        self.last_col, self.last_val = col, val

    def submit(self) -> None:
        values = values_from_columns(self.cols)
        if values is None:
            return
        self.locked, self.levels = True, []
        self._gen = iter_levels(values)        # filled one level per frame so progress is visible

    # ---------------------------------------------------------------- events
    def handle_event(self, e: pygame.event.Event) -> bool:
        """Returns False when the window should close."""
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.submit_rect.collidepoint(e.pos) and self.can_submit():
                self.submit()
            elif self.redraw_rect.collidepoint(e.pos):
                self.reset()
            elif self.normal_rect.inflate(150, 10).collidepoint(e.pos):
                self.show_normal = not self.show_normal
            elif self.orig_rect.inflate(160, 10).collidepoint(e.pos):
                self.show_original = not self.show_original
            elif self.ready and abs(e.pos[1] - self.rail_y) < 34 and self.rail_xs[0] - 20 <= e.pos[0] <= self.rail_xs[-1] + 20:
                self.sel = int(np.argmin(np.abs(self.rail_xs - e.pos[0])))
            elif not self.locked and self.x0 - 10 <= e.pos[0] <= self.x1 + 10 and PLOT.top <= e.pos[1] <= PLOT.bottom:
                self.drawing, self.last_col = True, None
                self._paint(e.pos)
        elif e.type == pygame.MOUSEMOTION and self.drawing:
            self._paint(e.pos)
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.drawing, self.last_col = False, None
        elif e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and self.can_submit():
                self.submit()
            elif e.key in (pygame.K_ESCAPE, pygame.K_r):
                self.reset()
            elif self.ready and e.key == pygame.K_RIGHT:
                self.sel = min(self.sel + 1, LEVELS - 1)
            elif self.ready and e.key == pygame.K_LEFT:
                self.sel = max(self.sel - 1, 0)
        return True

    def update(self) -> None:
        if self._gen is not None:               # one level per frame
            try:
                self.levels.append(next(self._gen))
            except StopIteration:
                self._gen = None
        if self.ready:
            target = self.levels[self.sel + 1].density
            ymax_t = max(0.45, 1.15 * float(target.max()))
            if self.shown is None:
                self.shown, self.shown_ymax = target.copy(), ymax_t
            else:                               # ease toward the newly selected density
                self.shown += (target - self.shown) * 0.35
                self.shown_ymax += (ymax_t - self.shown_ymax) * 0.35

    # --------------------------------------------------------------- drawing
    def _text(self, font, text, pos, color=TEXT, anchor="topleft"):
        surf = font.render(text, True, color)
        rect = surf.get_rect(**{anchor: pos})
        self.screen.blit(surf, rect)
        return rect

    def _button(self, rect, label, enabled=True, primary=False) -> None:
        fill = ACCENT if primary and enabled else PANEL
        border = ACCENT if primary and enabled else (RULE if not enabled else SOFT)
        color = INK if primary and enabled else (TEXT if enabled else (90, 98, 112))
        pygame.draw.rect(self.screen, fill, rect, border_radius=22)
        pygame.draw.rect(self.screen, border, rect, width=2, border_radius=22)
        self._text(self.f_body, label, rect.center, color, "center")

    def _checkbox(self, rect, label, on) -> None:
        pygame.draw.rect(self.screen, SOFT, rect, width=2, border_radius=5)
        if on:
            pygame.draw.rect(self.screen, ACCENT, rect.inflate(-8, -8), border_radius=3)
        self._text(self.f_body, label, (rect.right + 12, rect.centery), TEXT, "midleft")

    def _poly(self, pts, color, alpha, closed_to=None) -> None:
        if closed_to is not None:
            pts = [(pts[0][0], closed_to)] + list(pts) + [(pts[-1][0], closed_to)]
        layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.polygon(layer, (*color, alpha), pts)
        self.screen.blit(layer, (0, 0))

    def _dashed(self, pts, color, width=2, dash=7, gap=6) -> None:
        run, on = 0.0, True
        for a, b in zip(pts[:-1], pts[1:]):
            seg = math.dist(a, b)
            if on:
                pygame.draw.line(self.screen, color, a, b, width)
            run += seg
            if run >= (dash if on else gap):
                run, on = 0.0, not on

    def draw(self) -> None:
        s = self.screen
        s.fill(INK)
        self._text(self.f_title, TITLE, (40, 26))
        for i, line in enumerate(_wrap(self.f_body, DESCRIPTION, 1020)):
            self._text(self.f_body, line, (40, 72 + 24 * i), SOFT)

        pygame.draw.rect(s, PANEL, PLOT, border_radius=20)
        pygame.draw.rect(s, RULE, PLOT, width=1, border_radius=20)
        pygame.draw.line(s, SOFT, (self.x0, self.base), (self.x1, self.base), 1)

        if not self.locked:
            self._draw_drawing_phase()
        else:
            self._draw_locked_phase()

        self._button(self.submit_rect, "Submit curve", self.can_submit(), primary=True)
        self._button(self.redraw_rect, "Redraw", True)
        self._checkbox(self.normal_rect, "Normal curve", self.show_normal)
        self._checkbox(self.orig_rect, "Original curve", self.show_original)

        self._text(self.f_body, "Number of draws added together, n", (40, 596), TEXT)
        self._draw_rail()
        self._draw_chart()

        for i, line in enumerate(_wrap(self.f_small, CREDIT, 1020)):
            self._text(self.f_small, line, (40, 752 + 18 * i), SOFT)
        pygame.display.flip()

    def _draw_drawing_phase(self) -> None:
        drawn = np.flatnonzero(~np.isnan(self.cols))
        if drawn.size == 0:
            self._text(self.f_body, "Drag to draw a curve above the baseline. It becomes your probability density.",
                       PLOT.center, SOFT, "center")
            return
        xs = self.x0 + drawn / (RES - 1) * (self.x1 - self.x0)
        ys = self.base - np.clip(self.cols[drawn], 0, None) * (self.base - self.top)
        pts = list(zip(xs, ys))
        if len(pts) > 1:
            self.screen.set_clip(pygame.Rect(self.x0 - 12, PLOT.top + 40, self.x1 - self.x0 + 24, self.base - PLOT.top - 39))
            self._poly(pts, LILAC, 55, closed_to=self.base)
            pygame.draw.lines(self.screen, TEXT, False, pts, 3)
            self.screen.set_clip(None)
        self._text(self.f_small, "Press Submit curve (or Enter) to lock it in.", (PLOT.left + 24, PLOT.top + 16), SOFT)

    def _z_to_px(self, z):
        return self.x0 + (np.asarray(z) + Z_MAX) / (2 * Z_MAX) * (self.x1 - self.x0)

    def _draw_locked_phase(self) -> None:
        for z in range(-4, 5):                                        # axis ticks
            px = float(self._z_to_px(z))
            pygame.draw.line(self.screen, SOFT, (px, self.base), (px, self.base + 5), 1)
            self._text(self.f_small, f"{z}", (px, self.base + 8), SOFT, "midtop")
        self._text(self.f_small, "z : standard deviations from the mean", (self.x1, self.base + 24), SOFT, "topright")

        if not self.ready:
            done = max(len(self.levels) - 1, 0)
            self._text(self.f_body, f"Precomputing sums by doubling…  n = {2 ** max(done, 1) if done else 1}",
                       (PLOT.left + 24, PLOT.top + 16), SOFT)
            return

        ymax = self.shown_ymax
        to_y = lambda d: self.base - np.clip(d, 0, None) / ymax * (self.base - self.top)
        xs = self._z_to_px(Z_GRID)
        self.screen.set_clip(pygame.Rect(self.x0 - 2, PLOT.top + 40, self.x1 - self.x0 + 4, self.base - PLOT.top - 39))
        if self.show_original:                       # tall spikes in the original simply run off the top
            self._dashed(list(zip(xs, to_y(self.levels[0].density))), LILAC, 2)
        if self.show_normal:
            self._dashed(list(zip(xs, to_y(normal_pdf(Z_GRID)))), TEXT, 2)
        pts = list(zip(xs, to_y(self.shown)))
        self._poly(pts, ACCENT, 70, closed_to=self.base)
        pygame.draw.lines(self.screen, ACCENT, False, pts, 3)
        self.screen.set_clip(None)

        lv = self.levels[self.sel + 1]
        self._text(self.f_small, f"Sum of {lv.n} draws, rescaled to mean 0 and spread 1",
                   (PLOT.left + 24, PLOT.top + 16), SOFT)
        self._text(self.f_mono, f"n = {lv.n}   D = {lv.ks:.4f}   skew = {lv.skew:+.3f}   ex. kurtosis = {lv.excess_kurtosis:+.3f}",
                   (PLOT.right - 24, PLOT.top + 16), TEXT, "topright")

    def _draw_rail(self) -> None:
        pygame.draw.line(self.screen, RULE, (self.rail_xs[0], self.rail_y), (self.rail_xs[-1], self.rail_y), 4)
        for i, x in enumerate(self.rail_xs):
            on = self.ready
            pygame.draw.line(self.screen, SOFT if on else RULE, (x, self.rail_y - 9), (x, self.rail_y + 9), 2)
            self._text(self.f_mono, str(N_VALUES[i]), (x, self.rail_y + 16), TEXT if on and i == self.sel else (SOFT if on else (90, 98, 112)), "midtop")
        if self.ready:
            pygame.draw.circle(self.screen, ACCENT, (int(self.rail_xs[self.sel]), self.rail_y), 11)
            pygame.draw.circle(self.screen, INK, (int(self.rail_xs[self.sel]), self.rail_y), 4)
        else:
            self._text(self.f_small, "Submit a curve to unlock the ticks. Then click a tick, or use the left and right arrow keys.",
                       (40, 690), SOFT)

    def _draw_chart(self) -> None:
        r = self.chart_rect
        pygame.draw.rect(self.screen, PANEL, r, border_radius=16)
        pygame.draw.rect(self.screen, RULE, r, width=1, border_radius=16)
        self._text(self.f_small, "Distance to the normal curve (log scale)", (r.left + 14, r.top + 10), SOFT)
        if len(self.levels) < 3:               # need two points to draw a line
            return
        ks = np.array([lv.ks for lv in self.levels[1:]])
        lo, hi = ks.min() * 0.6, ks.max() * 1.5
        px0, px1, py0, py1 = r.left + 30, r.right - 24, r.bottom - 26, r.top + 40
        X = lambda i: px0 + i / (LEVELS - 1) * (px1 - px0)
        Y = lambda v: py0 - (math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) * (py0 - py1)
        ref = [(X(i), Y(ks[0] * 2 ** (-i / 2))) for i in range(LEVELS)]                    # 1 / sqrt(n) guide
        self._dashed(ref, SOFT, 1, 4, 5)
        pts = [(X(i), Y(v)) for i, v in enumerate(ks)]
        pygame.draw.lines(self.screen, LILAC, False, pts, 2)
        for i, p in enumerate(pts):
            pygame.draw.circle(self.screen, ACCENT if i == self.sel and self.ready else LILAC, p, 6 if i == self.sel and self.ready else 3)
        self._text(self.f_small, "2", (px0, r.bottom - 22), SOFT, "midtop")
        self._text(self.f_small, "1024", (px1, r.bottom - 22), SOFT, "midtop")
        self._text(self.f_small, "dashed: 1/\u221an", (r.right - 14, r.top + 10), SOFT, "topright")

    # ------------------------------------------------------------------ loop
    def run(self) -> None:
        running = True
        while running:
            for e in pygame.event.get():
                running = self.handle_event(e) and running
            self.update()
            self.draw()
            self.clock.tick(60)
        pygame.quit()


def main() -> None:
    App().run()
