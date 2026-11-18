"""0단계(권장): 탐지기·파이프라인 정상 동작 확인용 매니페스트.

AASIST가 학습된 데이터와 같은 계열인 ASVspoof 2019 LA 평가셋 일부로 채점해,
논문 수치(AASIST EER 약 0.83%)와 비슷하게 나오는지 확인한다.
여기서 수치가 크게 어긋나면 한국어 결과도 믿을 수 없으므로 먼저 원인을 찾는다.

데이터: https://datashare.ed.ac.uk/handle/10283/3336 의 LA.zip (약 7.6GB)

사용 예)
  python 00_asvspoof_manifest.py --la_dir D:/data/LA --out data/asv19 --n_bona 500 --n_per_attack 60
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from vfp.common import write_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--la_dir", required=True, help="LA.zip을 푼 폴더 (ASVspoof2019_LA_eval 등이 들어 있음)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_bona", type=int, default=500)
    ap.add_argument("--n_per_attack", type=int, default=60, help="공격 A07~A19 각각에서 뽑을 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    la = Path(args.la_dir)
    proto = la / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
    flac_dir = la / "ASVspoof2019_LA_eval" / "flac"
    groups = defaultdict(list)
    for line in proto.read_text().splitlines():
        spk, utt, _, attack, key = line.split()
        groups["real" if key == "bonafide" else attack].append((spk, utt))

    rng = random.Random(args.seed)
    rows = []
    for cond, items in sorted(groups.items()):
        rng.shuffle(items)
        n = args.n_bona if cond == "real" else args.n_per_attack
        for spk, utt in items[:n]:
            rows.append(dict(utt_id=utt, path=str(flac_dir / f"{utt}.flac"), lang="asv19",
                             condition=cond, label=1 if cond == "real" else 0,
                             source_utt=utt, speaker=spk, dur_sec=""))
    write_manifest(Path(args.out) / "manifest.csv", rows)
    print(f"완료: {len(rows)}개 (진짜 {sum(r['label'] for r in rows)}개) -> {args.out}/manifest.csv")


if __name__ == "__main__":
    main()
