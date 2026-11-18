"""정확도 검증: 원저자 공개 예제(ASVspoof 2019 LA 일부)로 3종이 '제대로' 동작하는지 확인.

왜 필요한가
  check_env.py는 '돌아가는지'만 봄. 가중치 변환이 조금만 어긋나도 모델은 돌아가지만 점수는 엉망이 됨.
  → 세 모델 모두 이 데이터와 같은 계열(ASVspoof 2019 LA)로 학습되었으므로 낮은 EER이 나와야 정상.

기대값
  AntiDeepfake-MMS300M : EER 약 6% (원저자 README에 같은 150개로 6.11% 보고)
  AASIST, W2V-AASIST   : EER 수 % 이하 (원 논문의 ASVspoof 2019 LA 전체 EER 1% 내외, 150개라 오차 큼)
  판정 기준: EER 15% 미만이면 정상, 30% 이상이면 변환·전처리 오류 의심 (무작위는 50%)

  python sanity_toy.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from vfp.common import load_audio, pick_device
from vfp.detectors import DEFAULT_SET, load


def eer(bona: np.ndarray, spoof: np.ndarray) -> float:
    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    frr = np.concatenate([[0], np.cumsum(labels)]) / len(bona)
    far = 1 - np.concatenate([[0], np.cumsum(1 - labels)]) / len(spoof)
    i = np.argmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2)


def read_toy(root: Path):
    files = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in (".wav", ".flac")}
    items = []
    for line in (root / "protocol.txt").read_text().split("\n"):
        parts = line.split()
        if len(parts) >= 5 and parts[1] in files and parts[1].startswith("LA_E_"):  # 평가(eval) 부분만
            items.append((files[parts[1]], 1 if parts[4] == "bonafide" else 0))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy_dir", default="data/toy_example")
    ap.add_argument("--weights_dir", default="weights")
    ap.add_argument("--detectors", default=DEFAULT_SET)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    device = pick_device(a.device)
    items = read_toy(Path(a.toy_dir))
    if not items:
        raise SystemExit(f"{a.toy_dir}에서 예제를 찾지 못함 → python setup_weights.py --only toy")
    n_b = sum(l for _, l in items)
    print(f"예제 {len(items)}개 (진짜 {n_b}, 가짜 {len(items) - n_b}), 장치 {device}")
    audio = [(load_audio(p), l) for p, l in items]

    for name in a.detectors.split(","):
        det = load(name, a.weights_dir, device)
        logits = np.stack([det.native(y) for y, _ in audio])
        s = logits[:, 1] - logits[:, 0]  # llr, 클수록 진짜
        lab = np.array([l for _, l in audio])
        e = eer(s[lab == 1], s[lab == 0]) * 100
        verdict = "정상" if e < 15 else ("오류 의심" if e >= 30 else "확인 필요")
        print(f"{name:22s} EER {e:5.2f}%  → {verdict}")
        del det


if __name__ == "__main__":
    main()
