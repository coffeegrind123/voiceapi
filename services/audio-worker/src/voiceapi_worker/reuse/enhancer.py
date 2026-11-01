"""RE-USE speech enhancement: STFT front end, chunked overlap-add, sweep-artifact guard.

Mirrors ``inference.py`` / ``inference_chunk.py`` from the model repository:
STFT sizes scale with the input rate from an 8 kHz base (n_fft 320, hop 40),
magnitudes are log1p-compressed, frames whose enhanced magnitude is mostly
exact zero are muted (upstream's "strange sweep artifact" fix), and inputs
longer than 30 s (upstream's guidance: training clips are < 10 s, long files
can run out of memory) run as 5 s Hann-weighted chunks at 50 % overlap.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from voiceapi_worker.reuse.model import SEMamba
from voiceapi_worker.reuse.scan import ScanBackend, available_backend

REPO_ID = "nvidia/NVIDIA-NemotronLabs-RE-USE"
REVISION = "31f82f2efad1095034492f236e8dac9239ba713b"
SUPPORTED_RATES = (8_000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000)
CHUNK_SECONDS = 5.0
HOP_PORTION = 0.5
WHOLE_FILE_MAX_SECONDS = 30.0
_RELU = nn.ReLU()


def _make_even(value: float) -> int:
    rounded = int(round(value))
    return rounded if rounded % 2 == 0 else rounded + 1


@dataclass(frozen=True, slots=True)
class StftSizes:
    n_fft: int
    hop: int
    win: int

    @classmethod
    def for_rate(cls, cfg: dict, sample_rate: int) -> StftSizes:
        stft = cfg["stft_cfg"]
        base = stft["sampling_rate"]
        return cls(
            n_fft=_make_even(stft["n_fft"] * sample_rate // base),
            hop=_make_even(stft["hop_size"] * sample_rate // base),
            win=_make_even(stft["win_size"] * sample_rate // base),
        )


def _stft(wav: torch.Tensor, sizes: StftSizes) -> tuple[torch.Tensor, torch.Tensor]:
    window = torch.hann_window(sizes.win, device=wav.device)
    spec = torch.stft(wav, sizes.n_fft, hop_length=sizes.hop, win_length=sizes.win, window=window, center=True,
                      pad_mode="reflect", normalized=False, return_complex=True)
    return torch.log1p(torch.abs(spec)), torch.angle(spec)


def _istft(mag: torch.Tensor, pha: torch.Tensor, sizes: StftSizes) -> torch.Tensor:
    mag = torch.expm1(_RELU(mag))
    spec = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    window = torch.hann_window(sizes.win, device=spec.device)
    return torch.istft(spec, sizes.n_fft, hop_length=sizes.hop, win_length=sizes.win, window=window, center=True)


def _match_length(reference: torch.Tensor, target: torch.Tensor, pad_value: float = 1e-8) -> torch.Tensor:
    length = reference.shape[1]
    if target.shape[1] >= length:
        return target[:, :length]
    padded = torch.full((target.shape[0], length), pad_value, dtype=target.dtype, device=target.device)
    padded[:, : target.shape[1]] = target
    return padded


class ReuseEnhancer:
    def __init__(self, model: SEMamba, cfg: dict, device: torch.device) -> None:
        self.model = model.to(device).eval()
        self.cfg = cfg
        self.device = device
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls, device: torch.device, *, cache_dir: str | None = None, local_dir: Path | None = None
    ) -> ReuseEnhancer:
        from safetensors.torch import load_file

        if local_dir is not None:
            config_path, weights_path = local_dir / "config.json", local_dir / "model.safetensors"
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(REPO_ID, "config.json", revision=REVISION, cache_dir=cache_dir))
            weights_path = Path(hf_hub_download(REPO_ID, "model.safetensors", revision=REVISION, cache_dir=cache_dir))
        cfg = json.loads(config_path.read_text())
        model = SEMamba(cfg)
        model.load_state_dict(load_file(str(weights_path)), strict=True)
        return cls(model, cfg, device)

    @property
    def scan_backend(self) -> ScanBackend:
        return available_backend(self.device)

    @torch.inference_mode()
    def _enhance_segment(self, wav: torch.Tensor, sizes: StftSizes) -> torch.Tensor:
        mag, pha = _stft(wav, sizes)
        amp_g, pha_g, _ = self.model(mag, pha)
        zero_portion = torch.sum(torch.expm1(_RELU(amp_g)) == 0, 1) / amp_g.shape[1]
        amp_g[:, :, (zero_portion > 0.5)[0]] = 0
        return _match_length(wav, _istft(amp_g, pha_g, sizes))

    @torch.inference_mode()
    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Enhance mono float32 audio at one of the supported rates; returns audio at the same rate."""
        if sample_rate not in SUPPORTED_RATES:
            raise ValueError(f"RE-USE supports {SUPPORTED_RATES} Hz input, got {sample_rate}")
        sizes = StftSizes.for_rate(self.cfg, sample_rate)
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).to(self.device).unsqueeze(0)
        with self._lock:
            if wav.shape[1] <= int(WHOLE_FILE_MAX_SECONDS * sample_rate):
                out = self._enhance_segment(wav, sizes)
            else:
                out = self._overlap_add(wav, sizes, int(CHUNK_SECONDS * sample_rate))
        return out[0].float().cpu().numpy()

    def _overlap_add(self, wav: torch.Tensor, sizes: StftSizes, chunk: int) -> torch.Tensor:
        hop = int(HOP_PORTION * chunk)
        window = torch.clamp(torch.hann_window(chunk, device=self.device), min=1e-7)
        enhanced = torch.zeros_like(wav)
        weight = torch.zeros_like(wav)
        count = max(1, math.ceil((wav.shape[1] - chunk) / hop) + 1)
        for index in range(count):
            if index < count - 1:
                start = index * hop
                piece = self._enhance_segment(wav[:, start : start + chunk], sizes)
                enhanced[:, start : start + chunk] += piece * window[: piece.shape[1]]
                weight[:, start : start + chunk] += window[: piece.shape[1]]
            else:
                piece = self._enhance_segment(wav[:, -chunk:], sizes)
                enhanced[:, -chunk:] += piece * window[: piece.shape[1]]
                weight[:, -chunk:] += window[: piece.shape[1]]
        return enhanced / weight


def snr_estimate_db(audio: np.ndarray, sample_rate: int) -> float:
    """Blind SNR: energy of the loudest 30 % of 20 ms frames over the quietest 30 %."""
    frame = max(1, int(0.02 * sample_rate))
    usable = len(audio) // frame * frame
    if usable < frame * 10:
        return float("nan")
    energy = np.mean(audio[:usable].reshape(-1, frame).astype(np.float64) ** 2, axis=1) + 1e-12
    ordered = np.sort(energy)
    cut = max(1, int(0.3 * len(ordered)))
    return float(10 * np.log10(ordered[-cut:].mean() / ordered[:cut].mean()))


__all__ = ["REVISION", "SUPPORTED_RATES", "ReuseEnhancer", "snr_estimate_db"]
