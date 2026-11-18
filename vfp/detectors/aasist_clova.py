"""① AASIST / AASIST-L (clovaai/aasist). 가중치는 저장소에 포함되어 있음."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import Detector


class ClovaAASIST(Detector):
    license_note = "MIT (NAVER Corp.)"

    def __init__(self, name: str, weights_dir: str, device: str = "cpu", **_):
        super().__init__(device)
        self.name = name
        repo = Path(weights_dir) / "aasist"
        if not (repo / "models" / "AASIST.py").exists():
            raise FileNotFoundError(f"{repo} 없음 → python setup_weights.py --only aasist 실행")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from models.AASIST import Model  # noqa: E402

        conf = json.loads((repo / "config" / f"{name}.conf").read_text())
        self.model = Model(conf["model_config"])
        state = torch.load(repo / "models" / "weights" / f"{name}.pth", map_location="cpu")
        self.model.load_state_dict(state)  # strict
        self.model.to(device).eval()

    @torch.no_grad()
    def logits(self, batch: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self.device)
        _, out = self.model(x)  # out[:, 0]=spoof, out[:, 1]=bona fide
        return out.float().cpu().numpy()
