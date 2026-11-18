# 딥보이스 탐지기 비교 키트 (deepvoice-bench)

공개 딥보이스 탐지 모델 3종을 같은 데이터·같은 조건에서 돌리기 위한 실행 환경임
(추진일정 2번: 모델 실행 환경 구축)

## 1. 비교 대상

| 이름 | 모델 | 크기 | 가중치 출처 | 라이선스 |
|---|---|---|---|---|
| `AASIST` | NAVER Clova AASIST (경량 기준선) | 30만 | clovaai/aasist 저장소 | MIT |
| `W2V-AASIST` | XLS-R 300M + AASIST | 3.2억 | SSL_Anti-spoofing (Google Drive) | MIT |
| `AntiDeepfake-MMS300M` | NII AntiDeepfake MMS-300M (다국어 대규모 학습) | 3.2억 | Zenodo / Hugging Face | 코드 BSD-3, 가중치 CC BY-NC-SA 4.0 |

- 선택 항목: `AASIST-L`(경량판), `W2V-AASIST-DF`(DF 평가용 가중치)
- 모델 선정 근거는 `공개_딥보이스_탐지모델_조사분석.docx` 참고

## 2. 설치

### 방법 A: 내 컴퓨터 (Windows·Linux·macOS 공통)

```
python -m venv .venv
.venv\Scripts\activate            (macOS/Linux: source .venv/bin/activate)
pip install -r requirements-bench.txt
```

- Python 3.10~3.13
- NVIDIA GPU가 있으면 pytorch.org 안내대로 CUDA 버전 torch를 **먼저** 설치
- AMD 라데온(Windows)은 CPU로 실행됨 → 추론은 가능, 미세조정은 방법 B 권장
- 한국어 데이터 준비·가짜 음성 생성까지 하려면 `requirements.txt`(전체) 설치

### 방법 B: Google Colab (GPU 무료)

- `colab_setup.ipynb`를 Colab에서 열고 위에서부터 실행
- 키트 zip은 Google Drive `MyDrive/deepvoice-bench/`에 올려 둠
- 가중치를 Drive에 저장해 다음 세션부터 재사용함

## 3. 실행 순서

| 순서 | 명령 | 확인할 것 |
|---|---|---|
| 1 | `python setup_weights.py` | 마지막 줄이 `aasist OK, ssl_aasist OK, antideepfake OK, toy OK` |
| 2 | `python check_env.py` | 3종 모두 `OK`, 판정 줄이 "3종 모두 정상 동작" |
| 3 | `python sanity_toy.py` | 3종 모두 `정상` (EER 15% 미만) |

- 2번은 '돌아가는지', 3번은 '맞게 돌아가는지'를 봄
  - 가중치 변환이 조금만 어긋나도 2번은 통과하지만 3번에서 EER 30~50%가 나옴
  - AntiDeepfake 기대값: 원저자 README에 같은 예제로 EER 6.11% 보고
- 3번까지 통과하면 실행 환경 구축 완료

## 4. 이후 단계에서 쓰는 스크립트 (기존 사전 실험 키트와 같은 흐름)

| 단계 | 명령 | 내용 |
|---|---|---|
| 0 | `00_asvspoof_manifest.py` | ASVspoof 2019 LA 목록 (선택, 대규모 재현 확인용) |
| 1 | `01_prepare_real.py` | 한국어 진짜 음성 목록 |
| 2 | `02_resynth.py` | 코덱·보코더 재합성 가짜 음성, `--telephony g711` 전화 음질 |
| 3 | `03_score.py --manifest ... --out ...` | **3종 채점** (기본값이 3종) |
| 4 | `04_analyze.py` | 탐지기별 EER·신뢰구간·그래프 |

- 처음에는 `03_score.py --limit 20`으로 빠른 점검
- 점수 열
  - `llr`: 각 원저자 평가 방식(native) 점수, 클수록 진짜
    - AASIST 계열: 앞 4초(짧으면 반복)
    - AntiDeepfake: 발화 전체(최대 30초), 파형 정규화 포함
  - `llr_multi`: 3종 공통 방식(4초 창을 2초 간격으로 밀며 평균) → 길이 처리 차이 없이 비교할 때 사용
  - `p_bona`: 0.5 기준 고정 판정용 확률
  - `sec_per_file`: 파일당 처리 시간

## 5. 기술 메모: fairseq 없이 실행하는 방식

- 문제
  - W2V-AASIST와 AntiDeepfake는 원래 fairseq 0.12로 모델을 만듦
  - fairseq는 Python 3.11 이상, Windows, 최신 Colab에서 설치가 사실상 불가능
- 해결
  - 같은 구조를 Hugging Face transformers `Wav2Vec2Model`로 만들고, fairseq 이름의 가중치를 변환해 넣음 (`vfp/detectors/w2v.py`)
  - W2V-AASIST의 후단(AASIST) 코드는 원본을 그대로 가져오고 생성자만 수정 (`vfp/detectors/vendor/`, MIT 고지 포함)
  - XLS-R 원본(1.2GB)은 필요 없음: 학습된 가중치 파일에 SSL 부분이 모두 들어 있음
- 검증 (`python -m tests.test_w2v_convert`)
  - torchaudio의 독립 구현(PyTorch 공식 fairseq 변환기)과 출력을 비교 → 오차 0
  - 일부러 틀린 변환(1개 층의 q·k 뒤바꿈)은 불일치로 검출됨을 확인
- 최종 확인은 실제 가중치로 `sanity_toy.py` 통과 여부

## 6. 검증 내역

- **실제 가중치 검증 완료 (2026-11-08, 데스크톱 RTX 2070, Windows 11, Python 3.14)**
  - check_env: 5개 탐지기 모두 정상 적재·동작, GPU에서 6초 음성 판정 0.02~0.07초
  - sanity_toy (원저자 예제 150개): AASIST 0.00%, W2V-AASIST 0.00%, AntiDeepfake **6.11%** → 원저자 보고값 6.11%와 정확히 일치
  - AntiDeepfake 가중치는 Zenodo 시간 초과로 Hugging Face 판(safetensors)으로 받아짐

- 클라우드 작업 공간(CPU, Python 3.13)에서 실제 실행으로 확인한 항목
  - fairseq → transformers 변환: torchaudio 독립 구현과 출력 일치 (위 5절)
  - AASIST·AASIST-L: 실제 가중치로 채점, 기존 사전 실험 키트와 점수 완전 일치
  - W2V-AASIST·AntiDeepfake: **원본과 같은 형식·같은 크기의 무작위 가중치 파일**로 적재(빠짐·남음 없음), 채점, 분석까지 전체 흐름
  - 처리 속도(CPU 2코어, 6초 음성 1개): AASIST 약 0.45초, W2V-AASIST 약 0.9초, AntiDeepfake 약 1.2초
  - 예제 압축 해제, 다운로드 실패 시 안내 문구
- 작업 공간에서 **확인하지 못한** 항목 (Google Drive·Zenodo·Hugging Face 접속 차단)
  - 실제 가중치 파일 다운로드와 그 파일 구조
    - W2V-AASIST: 원본 코드 기준으로 `ssl_model.model.*` + 후단 구조를 가정, 파일명이 다르면 이름으로 자동 탐색
    - AntiDeepfake: 원본 코드 기준으로 `m_ssl.model.*` + `proj_fc.*` 구조를 가정 (Hugging Face 파일은 접두어가 달라도 자동 처리)
  - → **첫 실행 때 `sanity_toy.py`로 반드시 확인**. 오류가 나면 오류 메시지를 그대로 공유

## 7. 문제 해결

| 증상 | 대응 |
|---|---|
| `가중치 적재 불일치` | 받은 파일이 다른 모델용일 가능성 → 파일 이름 확인, 오류 전문 공유 |
| Google Drive `Too many users` / 다운로드 실패 | 일일 한도 → 브라우저로 직접 받아 `weights/ssl_aasist/`에 넣기 |
| Zenodo 느림 | Hugging Face로 자동 전환됨, 둘 다 실패 시 브라우저로 받기 |
| `sanity_toy.py`에서 EER 30% 이상 | 변환·전처리 오류 의심 → 실행 환경 정보(`check_env.py` 출력)와 함께 공유 |
| CUDA 메모리 부족 | 3억 모델 추론은 GPU 4GB 이상이면 충분, `--device cpu`로도 가능 |

## 8. 라이선스

- 본 키트 코드: 팀 내부 사용
- 포함된 외부 코드: SSL_Anti-spoofing 후단(MIT, `vfp/detectors/vendor/` 상단에 원문 고지)
- 가중치는 키트에 포함하지 않음 (각 출처에서 직접 받음)
  - AntiDeepfake 가중치는 비상업 라이선스 → 졸업작품·연구 용도는 가능, 상용화 시 재검토
