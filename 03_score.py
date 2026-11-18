"""3단계: 공개 탐지기 3종으로 채점.

탐지기 (vfp/detectors 참고)
  AASIST                : 경량 기준선 (가중치 30만 개)
  W2V-AASIST            : XLS-R 300M + AASIST
  AntiDeepfake-MMS300M  : 다국어 대규모 사후학습 모델
  (선택) AASIST-L, W2V-AASIST-DF

점수 열 (04_analyze.py와 호환)
  logit_spoof, logit_bona : 모델 출력 (bona = 진짜)
  llr                     : logit_bona - logit_spoof, native 방식(각 원저자 평가 방식)
  p_bona                  : native 방식 softmax 확률. 0.5 이상이면 '진짜' 판정
  llr_multi               : win4 방식(4초 창 2초 간격 평균) → 세 모델 공통 조건 비교용
  sec_per_file            : 파일 1개 처리 시간(native 기준)

사용 예)
  python 03_score.py --manifest data/real_ko/manifest.csv --manifest data/fake_ko/manifest.csv \
      --out results/scores_ko.csv
  python 03_score.py ... --detectors AASIST --limit 20   # 빠른 점검
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from vfp.common import load_audio, pick_device, read_manifest
from vfp.detectors import DEFAULT_SET, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True, help="여러 번 지정 가능")
    ap.add_argument("--detectors", default=DEFAULT_SET)
    ap.add_argument("--weights_dir", default="weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="매니페스트마다 앞에서 N개만 (빠른 점검)")
    ap.add_argument("--skip_multi", action="store_true", help="win4 점수 생략(약 2배 빠름)")
    args = ap.parse_args()

    device = pick_device(args.device)
    rows = []
    for m in args.manifest:
        r = read_manifest(m)
        rows += r[: args.limit] if args.limit else r
    print(f"장치: {device}, 채점 대상 {len(rows)}개")
    audio = {r["path"]: load_audio(r["path"]) for r in rows}

    results = []
    for name in [d.strip() for d in args.detectors.split(",") if d.strip()]:
        t_load = time.time()
        det = load(name, args.weights_dir, device)
        print(f"[{name}] 불러오기 {time.time() - t_load:.1f}초, 파라미터 {det.n_params() / 1e6:.1f}M")
        t0 = time.time()
        for i, r in enumerate(rows):
            y = audio[r["path"]]
            t1 = time.time()
            o = det.native(y)
            dt = time.time() - t1
            m = float("nan") if args.skip_multi else det.win4(y)
            p = np.exp(o - o.max())
            p = p / p.sum()
            results.append({**r, "detector": name, "logit_spoof": f"{o[0]:.5f}", "logit_bona": f"{o[1]:.5f}",
                            "llr": f"{o[1] - o[0]:.5f}", "p_bona": f"{p[1]:.5f}", "llr_multi": f"{m:.5f}",
                            "sec_per_file": f"{dt:.3f}"})
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(rows)}")
        print(f"[{name}] {len(rows)}개 채점 {time.time() - t0:.0f}초")
        del det

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"완료 -> {args.out}")


if __name__ == "__main__":
    main()
