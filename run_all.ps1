# 전체 실행 스크립트 (Windows PowerShell)
# 사용: 아래 경로 변수만 본인 환경에 맞게 고친 뒤  .\run_all.ps1
# 빠른 점검만 하려면 $LIMIT = 20 으로 두고 먼저 돌려 볼 것 (0이면 전체)
# 처음 한 번: python setup_weights.py → python check_env.py → python sanity_toy.py

$EN_DIR   = "D:\data\LibriSpeech\test-clean"  # 영어 진짜 음성 폴더
$LA_DIR   = "D:\data\LA"                      # ASVspoof 2019 LA (없으면 빈 문자열 "")
$N        = 200
$LIMIT    = 0
$SYSTEMS  = "griffinlim,hifigan_16k,dac_16k,dac_16k_q4,encodec_6,encodec_1.5,mimi_q8,snac_24k"

$ErrorActionPreference = "Stop"
$lim = @(); if ($LIMIT -gt 0) { $lim = @("--limit", $LIMIT) }

# 0. 정상 동작 확인 (ASVspoof 2019 LA 일부)
if ($LA_DIR -ne "") {
  python 00_asvspoof_manifest.py --la_dir $LA_DIR --out data/asv19
  python 03_score.py --manifest data/asv19/manifest.csv --out results/scores_asv19.csv
}

# 1. 진짜 음성: 한국어(Zeroth-Korean, Hugging Face) / 영어(LibriSpeech, 로컬)
python 01_prepare_real.py --source hf --hf_name Bingsu/zeroth-korean --hf_split test --lang ko --out data/real_ko --n $N
python 01_prepare_real.py --source dir --input_dir $EN_DIR --speaker_depth 2 --lang en --out data/real_en --n $N

# 2. 가짜 음성(재합성) + 전화망 열화
python 02_resynth.py --real data/real_ko/manifest.csv --out data/fake_ko --systems $SYSTEMS --telephony g711 @lim
python 02_resynth.py --real data/real_en/manifest.csv --out data/fake_en --systems $SYSTEMS --telephony g711 @lim

# 3. 채점
python 03_score.py --manifest data/real_ko/manifest.csv --manifest data/fake_ko/manifest.csv --out results/scores_ko.csv
python 03_score.py --manifest data/real_en/manifest.csv --manifest data/fake_en/manifest.csv --out results/scores_en.csv

# 4. 분석
$scores = @("--scores", "results/scores_ko.csv", "--scores", "results/scores_en.csv")
if (Test-Path results/scores_asv19.csv) { $scores += @("--scores", "results/scores_asv19.csv") }
python 04_analyze.py @scores --out results
Write-Host "끝. results/report.md 를 열어 볼 것"
