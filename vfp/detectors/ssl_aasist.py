"""② Wav2Vec2-AASIST (TakHemlata/SSL_Anti-spoofing).

원본은 fairseq로 XLS-R 300M을 만든 뒤 학습된 가중치 전체(SSL 포함)를 덮어씀.
→ 본 키트는 transformers로 같은 구조를 만들고 fairseq 이름 가중치를 변환해 넣음 (w2v.py).
XLS-R 원본 체크포인트(xlsr2_300m.pt, 1.2GB)는 필요 없음: 학습된 가중치 파일에 SSL 부분이 모두 들어 있음.

가중치 파일 (Google Drive, 원저자 배포)
  W2V-AASIST    : LA_model.pth           (ASVspoof 2021 LA 평가용, 리더보드 수치의 기준)
  W2V-AASIST-DF : Best_LA_model_for_DF.pth (ASVspoof 2021 DF 평가용, 선택)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import Detector
from .w2v import build_from_checkpoint, load_any

FILES = {"W2V-AASIST": ("LA_model.pth",), "W2V-AASIST-DF": ("Best_LA_model_for_DF.pth",)}


class _SSL(nn.Module):
    """원본 SSLModel과 같은 역할: (B, T) 파형 → (B, 프레임, 1024)."""

    def __init__(self, hf_model):
        super().__init__()
        self.model = hf_model
        self.out_dim = hf_model.config.hidden_size

    def extract_feat(self, x):
        if x.ndim == 3:
            x = x[:, :, 0]
        return self.model(x).last_hidden_state


def find_weight(weights_dir: str, name: str) -> Path:
    d = Path(weights_dir) / "ssl_aasist"
    for fn in FILES[name]:
        if (d / fn).exists():
            return d / fn
    # 파일 이름이 바뀌었을 경우: DF 여부로 추정
    cands = sorted(d.rglob("*.pth"))
    want_df = name.endswith("DF")
    cands = [c for c in cands if ("DF" in c.name) == want_df]
    if cands:
        return cands[0]
    raise FileNotFoundError(f"{d}에 {FILES[name][0]} 없음 → python setup_weights.py --only ssl_aasist 실행")


class SSLAASIST(Detector):
    license_note = "MIT (Hemlata Tak)"

    def __init__(self, name: str, weights_dir: str, device: str = "cpu", **_):
        super().__init__(device)
        from .vendor.ssl_aasist_backend import Model

        self.name = name
        self.weight_path = find_weight(weights_dir, name)
        state = load_any(self.weight_path)
        hf, prefix = build_from_checkpoint(state)
        if prefix != "ssl_model.model.":
            raise RuntimeError(f"예상과 다른 체크포인트 구조(접두어 {prefix})")
        self.model = Model(_SSL(hf), device)
        rest = {k: v for k, v in state.items() if not k.startswith(prefix)}
        missing, unexpected = self.model.load_state_dict(rest, strict=False)
        missing = [k for k in missing if not k.startswith("ssl_model.")]
        if missing or unexpected:
            raise RuntimeError(f"후단 가중치 불일치 — 빠짐 {missing[:5]} 남음 {unexpected[:5]}")
        self.model.to(device).eval()

    @torch.no_grad()
    def logits(self, batch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self.device)
        out = self.model(x)  # out[:, 1] = bona fide (원본 main_SSL_LA.py와 동일)
        return out.float().cpu().numpy()
