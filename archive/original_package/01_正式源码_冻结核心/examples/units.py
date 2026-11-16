"""Demonstration-only input interfaces; NOT part of DNO or a claim of perception.

All input methods return [N,d] numeric vectors to the unchanged dno.DNO.
The segmentation/embedding choice belongs to an application. In particular,
these examples DO NOT discover semantic objects, contacts or natural laws.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _compress(samples: Tensor, d: int) -> Tensor:
    """Fixed-size coarse numeric summary, never a learned hidden DNO rule."""
    if d < 1 or samples.ndim != 2 or not samples.size(0):
        raise ValueError('expect nonempty [N,P] input and d>0')
    # Adaptive 1D average pooling maps arbitrary flattened block length to d.
    # This is purely an EXTERNAL representation contract.
    return torch.nn.functional.adaptive_avg_pool1d(samples.unsqueeze(1), d).squeeze(1)


def image_units(image: Tensor, *, d: int = 8, patch: int = 4) -> Tensor:
    """Non-overlapping 2D patches -> numeric [N,d]. Supports gray or HWC."""
    if image.ndim not in (2, 3):
        raise ValueError('image must be [H,W] or [H,W,C]')
    if patch <= 0:
        raise ValueError('patch must be positive')
    x = image.to(torch.float32)
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    h, w, c = x.shape
    if h % patch or w % patch or not h or not w:
        raise ValueError('height/width must be positive multiples of patch')
    blocks = x.reshape(h//patch,patch,w//patch,patch,c).permute(0,2,1,3,4)
    return _compress(blocks.reshape(-1,patch*patch*c),d)


def video_units(video: Tensor, *, d: int = 8, frames: int = 2,
                patch: int = 4) -> Tensor:
    """Contiguous spatiotemporal cubes -> numeric [N,d], not a video model."""
    if video.ndim not in (3, 4):
        raise ValueError('video must be [T,H,W] or [T,H,W,C]')
    if frames <= 0 or patch <= 0:
        raise ValueError('frame group and patch must be positive')
    x = video.to(torch.float32)
    if x.ndim == 3:
        x = x.unsqueeze(-1)
    t,h,w,c = x.shape
    if not t or not h or not w or t % frames or h % patch or w % patch:
        raise ValueError('T/H/W must be positive multiples of block sizes')
    blocks = x.reshape(t//frames,frames,h//patch,patch,w//patch,patch,c)
    blocks = blocks.permute(0,2,4,1,3,5,6)
    return _compress(blocks.reshape(-1,frames*patch*patch*c),d)


def audio_units(audio: Tensor, *, d: int = 8, frame: int = 64) -> Tensor:
    """Time windows -> log magnitude spectrum [N,d]; no voice recognition."""
    if audio.ndim != 1 or not audio.numel() or frame <= 1 or audio.numel() % frame:
        raise ValueError('audio must be nonempty 1D, divisible by frame>1')
    chunks = audio.to(torch.float32).reshape(-1,frame)
    spectrum = torch.log1p(torch.fft.rfft(chunks, dim=1).abs())
    return _compress(spectrum, d)


def vector_units(vectors: Tensor, *, d: int = 8) -> Tensor:
    """For pre-embedded symbolic or sensor units; no tokenizer is required."""
    if vectors.ndim != 2 or vectors.shape[1] != d:
        raise ValueError('vectors must be [N,d]')
    return vectors.to(torch.float32)
