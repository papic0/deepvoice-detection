"""실행 환경 점검 + 탐지기별 동작 확인(스모크 테스트).

  python check_env.py              # 환경 정보 + 받아 둔 탐지기 전부 시험
  python check_env.py --device cpu

확인 항목
  - Python / PyTorch / transformers 버전, 사용 장치(GPU 이름·메모리), ffmpeg
  - 탐지기마다: 가중치 적재(빠짐·남음 없이), 6초 합성 음성 판정, 출력이 유한한 값인지, 처리 시간
결과 판정
  모두 OK면 실행 환경 구축 완료. 정확도 확인은 sanity_toy.py에서 함.
"""
from __future__ import annotations

import argparse
import platform
import shutil
import sys
import time
import traceback

import numpy as np


def env_report(device: str) -> None:
    import torch
    import transformers

    print("== 환경 ==")
    print(f"OS          : {platform.system()} {platform.release()}")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"transformers: {transformers.__version__}")
    print(f"장치        : {device}", end="")
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f" ({p.name}, {p.total_memory / 2**30:.1f}GB)")
    else:
        print()
    print(f"ffmpeg      : {'있음' if shutil.which('ffmpeg') else '없음 (전화 음질 조건 만들 때만 필요)'}")
    print()


def test_audio(sec: float = 6.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(sec * sr)) / sr
    rng = np.random.default_rng(0)
    y = 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) + 0.02 * rng.standard_normal(len(t))
    return y.astype(np.float32)


def main():
    from vfp.common import pick_device
    from vfp.detectors import REGISTRY, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights_dir", default="weights")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--detectors", default=",".join(REGISTRY))
    a = ap.parse_args()
    device = pick_device(a.device)
    env_report(device)

    y = test_audio()
    print("== 탐지기 ==")
    rows = []
    for name in a.detectors.split(","):
        try:
            t0 = time.time()
            det = load(name, a.weights_dir, device)
            t_load = time.time() - t0
            det.native(y)  # 예열(첫 호출은 느림)
            t1 = time.time()
            o = det.native(y)
            t_nat = time.time() - t1
            t2 = time.time()
            m = det.win4(y)
            t_win = time.time() - t2
            assert o.shape == (2,) and np.all(np.isfinite(o)) and np.isfinite(m), "출력 이상"
            rows.append((name, "OK", f"{det.n_params() / 1e6:.1f}M", f"{t_load:.1f}s", f"{t_nat:.2f}s", f"{t_win:.2f}s", det.license_note))
            del det
        except FileNotFoundError as e:
            rows.append((name, "가중치 없음", "-", "-", "-", "-", str(e).split("→")[-1].strip()))
        except Exception as e:
            traceback.print_exc()
            rows.append((name, "오류", "-", "-", "-", "-", f"{type(e).__name__}: {e}"[:80]))
    head = ("탐지기", "상태", "파라미터", "적재", "6초 판정", "4초창 평균", "비고")
    widths = [max(len(str(r[i])) for r in rows + [head]) for i in range(len(head))]
    for r in [head] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    core = [r for r in rows if r[0] in ("AASIST", "W2V-AASIST", "AntiDeepfake-MMS300M")]
    ok = all(r[1] == "OK" for r in core)
    print("\n판정:", "비교 대상 3종 모두 정상 동작 → 다음: python sanity_toy.py" if ok else "3종 중 준비 안 된 항목 있음 (위 비고 참고)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
