"""③ AntiDeepfake MMS-300M (nii-yamagishilab/AntiDeepfake).

구조 (원본 models/W2V.py): wav2vec2(MMS-300M) → 프레임 평균 → Linear(1024, 2)
  출력 index 1 = real (원본 dataio.py: real → 1)
입력 전처리 (원본 dataio.py): 16kHz 모노, 파형 전체에 layer_norm (평균 0, 분산 1)

가중치 (둘 중 하나)
  Zenodo      : mms_300m.ckpt          https://zenodo.org/records/15580543
  Hugging Face: model.safetensors      nii-yamagishilab/mms-300m-anti-deepfake
라이선스: 코드 BSD-3, 가중치 CC BY-NC-SA 4.0 (비상업·동일조건)

검증용 기대값 (원저자 README): 원저자 toy_example 150개에서 EER 약 6.1%, AUC 약 0.99
  → python sanity_toy.py 로 확인
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import NB_SAMP, Detector
from .w2v import build_from_checkpoint, load_any

SR = 16000


class _Model(nn.Module):
    def __init__(self, ssl, out_dim: int):
        super().__init__()
        self.ssl = ssl
        self.proj_fc = nn.Linear(out_dim, 2)

    def forward(self, x):
        emb = self.ssl(x).last_hidden_state  # (B, frame, dim)
        return self.proj_fc(emb.mean(dim=1))  # AdaptiveAvgPool1d(1)과 동일


def find_weight(weights_dir: str) -> Path:
    d = Path(weights_dir) / "antideepfake"
    for pat in ("mms_300m.ckpt", "*.safetensors", "*.ckpt", "*.pt", "*.bin"):
        c = sorted(d.rglob(pat))
        if c:
            return c[0]
    raise FileNotFoundError(f"{d}에 가중치 없음 → python setup_weights.py --only antideepfake 실행")


class AntiDeepfake(Detector):
    license_note = "코드 BSD-3 / 가중치 CC BY-NC-SA 4.0 (NII Yamagishi Lab)"

    def __init__(self, name: str, weights_dir: str, device: str = "cpu", max_sec: float = 30.0, **_):
        super().__init__(device)
        self.name = name
        self.max_len = int(max_sec * SR)
        self.weight_path = find_weight(weights_dir)
        state = load_any(self.weight_path)
        hf, _ = build_from_checkpoint(state)
        fc_w = [k for k in state if k.endswith("proj_fc.weight")]
        if len(fc_w) != 1:
            raise RuntimeError(f"분류층(proj_fc)을 찾지 못함: {fc_w}")
        pre = fc_w[0][: -len("weight")]
        self.model = _Model(hf, hf.config.hidden_size)
        self.model.proj_fc.load_state_dict({"weight": state[pre + "weight"], "bias": state[pre + "bias"]})
        if self.model.proj_fc.weight.shape != (2, hf.config.hidden_size):
            raise RuntimeError("분류층 크기 불일치")
        self.model.to(device).eval()

    @staticmethod
    def _norm(x: torch.Tensor) -> torch.Tensor:
        # 원본: torch.nn.functional.layer_norm(wav, wav.shape) — 파형마다 따로 정규화
        return torch.stack([F.layer_norm(r, r.shape) for r in x])

    @torch.no_grad()
    def logits(self, batch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self.device)
        out = self.model(self._norm(x))  # [:, 0]=fake, [:, 1]=real
        return out.float().cpu().numpy()

    def native(self, y: np.ndarray) -> np.ndarray:
        """발화 전체를 한 번에 판정 (원저자 평가 방식). 너무 길면 max_sec 단위로 나눠 logit 평균."""
        if len(y) < 400:  # 25ms 미만은 합성곱 특징 추출 불가
            y = np.tile(y, 400 // max(len(y), 1) + 1)
        if len(y) <= self.max_len:
            return self.logits(y[None])[0]
        segs = [y[s:s + self.max_len] for s in range(0, len(y), self.max_len)]
        segs = [s for s in segs if len(s) >= NB_SAMP] or segs[:1]
        return np.mean([self.logits(s[None])[0] for s in segs], axis=0)
