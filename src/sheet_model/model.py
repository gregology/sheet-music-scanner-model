"""The line reader: one straightened line of music in, a probability for every token at every step out.

It reads the whole line in one pass and has no loop, so it converts to Core ML as it is (README.md, "Where it runs").
A stack of convolutions turns the picture into 200 columns of features; a few layers of self-attention let each
column see the whole line, since a note's pitch depends on the clef and key at the line's start; and CTC turns the
columns into tokens.
"""

from __future__ import annotations

import torch
from torch import nn

from .lines import HEIGHT, WIDTH

STEPS = WIDTH // 8   # columns of features: the most tokens a line can have is fewer than half of this


def block(inputs: int, outputs: int, stride: tuple[int, int]) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False), nn.BatchNorm2d(outputs), nn.SiLU(),
                         nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False), nn.BatchNorm2d(outputs), nn.SiLU())


class Reader(nn.Module):
    def __init__(self, classes: int, width: int = 256, layers: int = 4, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.settings = {"classes": classes, "width": width, "layers": layers, "heads": heads, "dropout": dropout}
        self.picture = nn.Sequential(
            block(1, 32, (2, 2)), block(32, 64, (2, 2)), block(64, 128, (2, 2)), block(128, width, (2, 1)), block(width, width, (2, 1)),
            nn.Conv2d(width, width, (HEIGHT // 32, 1), bias=False), nn.BatchNorm2d(width), nn.SiLU(),   # what is left of the height, folded into each column
        )
        self.place = nn.Parameter(torch.zeros(1, STEPS, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.line = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, classes))
        nn.init.trunc_normal_(self.place, std=0.02)

    def forward(self, pictures: torch.Tensor) -> torch.Tensor:
        """`pictures`: batch by 1 by HEIGHT by WIDTH, paper near +1 and ink near -1. Returns batch by STEPS by classes,
        as scores before the softmax."""
        columns = self.picture(pictures).squeeze(2).transpose(1, 2)
        return self.out(self.line(columns + self.place))


def read(scores: torch.Tensor) -> list[tuple[list[int], list[float]]]:
    """The likeliest tokens of each line, by CTC's rule (drop repeats, then blanks), and how sure the reader was of each:
    the highest probability it gave the token over the steps that spelled it."""
    probabilities = scores.softmax(-1)
    sure, best = probabilities.max(-1)
    lines = []
    for line_best, line_sure in zip(best.tolist(), sure.tolist()):
        tokens, confidence, last = [], [], 0
        for token, certainty in zip(line_best, line_sure):
            if token != 0 and token != last:
                tokens.append(token)
                confidence.append(certainty)
            elif token != 0:
                confidence[-1] = max(confidence[-1], certainty)
            last = token
        lines.append((tokens, confidence))
    return lines
