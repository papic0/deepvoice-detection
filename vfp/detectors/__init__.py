"""탐지기 공통 인터페이스.

모든 탐지기는 16kHz 모노 파형을 받아 (logit_fake, logit_real) 두 값을 냄.
  llr    = logit_real - logit_fake   (클수록 '진짜')
  p_real = softmax(logits)[real]

채점 방식 두 가지
  native : 각 모델 원저자가 평가에 쓴 방식 그대로
           - AASIST 계열: 앞 4초(64,600샘플), 짧으면 반복해서 채움
           - AntiDeepfake: 발화 전체(최대 max_sec초, 넘으면 구간별 평균), 파형 정규화 포함
  win4   : 세 모델 공통 비교용. 4초 창을 2초 간격으로 밀며 llr 평균 → 길이 처리 방식 차이를 제거
"""
from __future__ import annotations

import numpy as np

NB_SAMP = 64600  # 약 4.04초 @16kHz
HOP = 32000      # 2초


def pad_repeat(x: np.ndarray, n: int = NB_SAMP) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    reps = n // max(len(x), 1) + 1
    return np.tile(x, reps)[:n]


def windows(x: np.ndarray, n: int = NB_SAMP, hop: int = HOP) -> list[np.ndarray]:
    if len(x) <= n:
        return [pad_repeat(x, n)]
    return [x[s:s + n] for s in range(0, len(x) - n + 1, hop)]


class Detector:
    name = "base"
    license_note = ""

    def __init__(self, device: str = "cpu"):
        self.device = device

    # (B, T) 같은 길이 배치 → (B, 2) [fake, real]
    def logits(self, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def native(self, y: np.ndarray) -> np.ndarray:
        return self.logits(pad_repeat(y)[None])[0]

    def win4(self, y: np.ndarray, batch: int = 8) -> float:
        ws = np.stack(windows(y))
        out = np.concatenate([self.logits(ws[i:i + batch]) for i in range(0, len(ws), batch)])
        return float(np.mean(out[:, 1] - out[:, 0]))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


REGISTRY = {
    # 이름: (모듈, 클래스, 설명)
    "AASIST": ("aasist_clova", "ClovaAASIST", "NAVER Clova AASIST, ASVspoof 2019 LA"),
    "AASIST-L": ("aasist_clova", "ClovaAASIST", "AASIST 경량판"),
    "W2V-AASIST": ("ssl_aasist", "SSLAASIST", "XLS-R 300M + AASIST (SSL_Anti-spoofing LA 가중치)"),
    "W2V-AASIST-DF": ("ssl_aasist", "SSLAASIST", "같은 구조, DF 평가용 가중치 (선택)"),
    "AntiDeepfake-MMS300M": ("antideepfake", "AntiDeepfake", "NII AntiDeepfake MMS-300M, 다국어 사후학습"),
}

DEFAULT_SET = "AASIST,W2V-AASIST,AntiDeepfake-MMS300M"


def load(name: str, weights_dir: str, device: str = "cpu", **kw) -> Detector:
    import importlib

    if name not in REGISTRY:
        raise KeyError(f"알 수 없는 탐지기 {name}. 사용 가능: {', '.join(REGISTRY)}")
    mod, cls, _ = REGISTRY[name]
    m = importlib.import_module(f"vfp.detectors.{mod}")
    return getattr(m, cls)(name=name, weights_dir=weights_dir, device=device, **kw)
