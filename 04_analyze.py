"""4단계: 분석 및 판정.

조건별로 '같은 언어의 진짜 음성' vs '해당 시스템의 가짜 음성'을 비교한다.

지표
  EER        : 오탐률=미탐률이 되는 지점의 오류율. 50%면 동전 던지기 수준(탐지 불가)
  95% CI     : 부트스트랩(진짜·가짜 각각 재표집) 신뢰구간
  미탐률@0.5 : p_bona>=0.5, 즉 가짜를 '진짜'로 통과시킨 비율 (실서비스 관점)
  오탐률@0.5 : 진짜를 '가짜'로 잘못 경고한 비율

사용 예)
  python 04_analyze.py --scores results/scores_ko.csv --scores results/scores_en.csv --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CODEC_PREFIX = ("dac", "encodec", "mimi", "snac")
CONTROL = ("griffinlim", "hifigan_16k")
TTS = ("xtts", "chatterbox", "omnivoice")  # 02b_tts.py 음성 복제 TTS
MIN_N = 50  # 이보다 적으면 수치 신뢰도 낮음 경고


def eer(bona: np.ndarray, spoof: np.ndarray) -> float:
    """점수가 클수록 진짜. 모든 임계값을 훑어 FAR=FRR 지점의 값을 반환."""
    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    # 임계값을 i번째 점수 '초과'로 둘 때: 진짜 중 기각(FRR), 가짜 중 수락(FAR)
    frr = np.concatenate([[0], np.cumsum(labels)]) / len(bona)
    far = 1 - np.concatenate([[0], np.cumsum(1 - labels)]) / len(spoof)
    i = np.argmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2)


def eer_ci(bona, spoof, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = [eer(rng.choice(bona, len(bona)), rng.choice(spoof, len(spoof))) for _ in range(n_boot)]
    return np.percentile(vals, [2.5, 97.5])


def system_type(cond: str) -> str:
    base = cond.split("|")[0]
    if base.startswith(CODEC_PREFIX):
        return "codec"
    if base in CONTROL:
        return "control"
    if base in TTS:
        return "tts"
    if base == "real":
        return "real"
    return "other"  # ASVspoof 공격 ID 등


def summarize(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = []
    for (det, lang), g in df.groupby(["detector", "lang"]):
        g = g.copy()
        g["channel"] = g["condition"].str.contains(r"\|g711").map({True: "g711", False: "clean"})
        for ch, gc in g.groupby("channel"):
            real = gc[gc["label"] == 1]
            if real.empty:
                continue
            bona = real[score_col].to_numpy(float)
            fa = float((real["p_bona"].astype(float) < 0.5).mean())
            fakes = gc[gc["label"] == 0]
            groups = list(fakes.groupby("condition"))
            if lang.startswith("asv"):  # 공식 데이터셋은 전체 묶음도 추가
                groups = [("ALL", fakes)] + groups
            for cond, gf in groups:
                spoof = gf[score_col].to_numpy(float)
                lo, hi = eer_ci(bona, spoof)
                out.append(dict(
                    detector=det, lang=lang, channel=ch, system=cond.split("|")[0],
                    type=system_type(cond), n_real=len(bona), n_fake=len(spoof),
                    eer=eer(bona, spoof), eer_lo=lo, eer_hi=hi,
                    miss_at_05=float((gf["p_bona"].astype(float) >= 0.5).mean()),
                    false_alarm_at_05=fa,
                ))
    return pd.DataFrame(out)


def verdict(s: pd.DataFrame, det: str = "AASIST") -> list[str]:
    lines = []
    k = s[(s.detector == det) & (s.lang == "ko") & (s.channel == "clean")]
    if k.empty:
        return ["- 한국어(ko) 결과가 없어 판정 불가"]
    codec = k[k.type == "codec"]
    ctrl = k[k.type == "control"]
    if codec.empty:
        return ["- 코덱 계열 결과가 없어 판정 불가"]
    c_eer = codec.eer.median()
    c_miss = codec.miss_at_05.median()
    b_eer = ctrl.eer.max() if not ctrl.empty else np.nan
    fa = k.false_alarm_at_05.iloc[0]
    lines.append(f"- 한국어 코덱 계열 EER 중앙값: {c_eer:.1%} (미탐률@0.5 중앙값 {c_miss:.1%})")
    if not np.isnan(b_eer):
        lines.append(f"- 한국어 대조군(보코더/고전) EER 최대값: {b_eer:.1%}")
    lines.append(f"- 한국어 진짜 음성 오탐률@0.5: {fa:.1%}")

    e = s[(s.detector == det) & (s.lang == "en") & (s.channel == "clean") & (s.type == "codec")]
    if not e.empty:
        lines.append(f"- 영어 코덱 계열 EER 중앙값: {e.eer.median():.1%} (원인 분리용 비교군)")

    ref = max(b_eer if not np.isnan(b_eer) else 0.0, 0.05)
    if c_eer >= 0.20 and c_eer >= 2 * ref:
        lines.append("- **판정: GO** — 기존 탐지기가 코덱 계열 가짜 음성을 대조군보다 뚜렷하게 놓침 → 주제 근거 성립")
    elif c_eer < 0.10 and fa < 0.30:
        lines.append("- **판정: NO-GO** — 기존 탐지기가 코덱 계열도 잘 잡음 → 이 방향의 차별성 약함, 주제 재검토")
    else:
        lines.append("- **판정: 보류** — 경계 구간. 표본 확대·강한 탐지기 추가 후 재확인 필요")
    if fa >= 0.30:
        lines.append("- 참고: 한국어 진짜 음성 오탐이 큼 → '언어 도메인 차이' 자체도 별도 근거가 됨")
    if not e.empty:
        en = e.eer.median()
        if en >= 0.20 and c_eer >= 0.20:
            lines.append("- 해석: 영어에서도 코덱 계열을 놓침 → 원인은 주로 '코덱 방식'(언어 무관)")
        elif en < 0.10 and c_eer >= 0.20:
            lines.append("- 해석: 영어는 잡고 한국어만 놓침 → 원인은 주로 '한국어(언어 차이)'")
    small = s[(s.n_real < MIN_N) | (s.n_fake < MIN_N)]
    if not small.empty:
        lines.append(f"- 주의: 표본 {MIN_N}개 미만 조건 있음 → 수치는 참고용(신뢰구간 확인)")
    return lines


def plot(s: pd.DataFrame, out: Path, det: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = s[(s.detector == det) & (s.channel == "clean") & (~s.lang.str.startswith("asv"))]
    if k.empty:
        return None
    systems = list(dict.fromkeys(k.sort_values(["type", "system"]).system))
    langs = sorted(k.lang.unique())
    width = 0.8 / len(langs)
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(systems)), 4))
    for j, lang in enumerate(langs):
        sub = k[k.lang == lang].set_index("system").reindex(systems)
        xs = np.arange(len(systems)) + j * width
        err = [sub.eer - sub.eer_lo, sub.eer_hi - sub.eer]
        ax.bar(xs, sub.eer * 100, width, yerr=np.array(err) * 100, capsize=3, label=lang.upper())
    ax.axhline(50, ls="--", lw=1, c="gray")
    ax.text(len(systems) - 0.5, 51, "chance (50%)", ha="right", va="bottom", fontsize=8, c="gray")
    ax.set_xticks(np.arange(len(systems)) + width * (len(langs) - 1) / 2)
    ax.set_xticklabels(systems, rotation=30, ha="right")
    ax.set_ylabel("EER (%)")
    ax.set_ylim(0, min(100, max(60, float(k.eer_hi.max()) * 100 + 5)))
    ax.set_title(f"{det}: EER by generation system (real vs fake, same language)")
    ax.legend()
    fig.tight_layout()
    path = out / f"eer_{det}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_scores(df: pd.DataFrame, out: Path, det: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df[(df.detector == det) & (~df.condition.str.contains(r"\|")) & (~df.lang.str.startswith("asv"))]
    if d.empty:
        return None
    conds = ["real"] + sorted(c for c in d.condition.unique() if c != "real")
    langs = sorted(d.lang.unique())
    fig, axes = plt.subplots(1, len(langs), figsize=(6 * len(langs), 4), squeeze=False)
    for ax, lang in zip(axes[0], langs):
        data = [d[(d.lang == lang) & (d.condition == c)].llr.astype(float).to_numpy() for c in conds]
        ax.boxplot([x if len(x) else [np.nan] for x in data], showfliers=False)
        ax.set_xticks(range(1, len(conds) + 1))
        ax.set_xticklabels(conds, rotation=30, ha="right")
        ax.axhline(0, ls="--", lw=1, c="gray")
        ax.set_title(f"{det} / {lang.upper()}: score (llr > 0 means judged REAL)")
        ax.set_ylabel("llr")
    fig.tight_layout()
    path = out / f"scores_{det}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(p) for p in args.scores], ignore_index=True)
    for c in ("llr", "llr_multi", "p_bona"):
        df[c] = df[c].astype(float)

    s = summarize(df, "llr")
    if df["llr_multi"].notna().all():
        sm = summarize(df, "llr_multi")[["detector", "lang", "channel", "system", "eer"]]
        s = s.merge(sm.rename(columns={"eer": "eer_multi"}),
                    on=["detector", "lang", "channel", "system"])
    else:  # 03단계를 --skip_multi로 돌린 경우
        s["eer_multi"] = np.nan
    s.to_csv(out / "summary.csv", index=False)

    md = ["# 사전 실험 결과", ""]
    for det in s.detector.unique():
        md += [f"## {det}", "", *verdict(s, det), ""]
        t = s[s.detector == det].copy()
        for c in ("eer", "eer_lo", "eer_hi", "eer_multi", "miss_at_05", "false_alarm_at_05"):
            t[c] = (t[c] * 100).round(1)
        t["eer_95ci"] = t.eer_lo.astype(str) + "~" + t.eer_hi.astype(str)
        cols = ["lang", "channel", "system", "type", "n_real", "n_fake", "eer", "eer_95ci",
                "eer_multi", "miss_at_05", "false_alarm_at_05"]
        md += [t[cols].to_markdown(index=False), ""]
        for p in (plot(s, out, det), plot_scores(df, out, det)):
            if p:
                md += [f"![{p.stem}]({p.name})", ""]
    md += ["## 지표 설명", "",
           "- eer: 진짜/가짜 구분 오류율(%) — 50에 가까울수록 탐지 불가",
           "- eer_95ci: 부트스트랩 95% 신뢰구간",
           "- eer_multi: 발화 전체를 4초 창으로 나눠 평균한 점수 기준 EER",
           "- miss_at_05: 가짜인데 p_bona>=0.5로 '진짜' 판정된 비율(%)",
           "- false_alarm_at_05: 진짜인데 '가짜' 판정된 비율(%)", ""]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:40]))
    print(f"\n완료 -> {out / 'report.md'}, {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
