"""명령을 실행하면서 화면 출력과 로그 파일(UTF-8)에 동시에 기록.

  python run_logged.py logs/step.txt <명령...>
  (명령이 .py로 시작하면 현재 파이썬으로 실행)
종료 코드는 명령의 종료 코드를 그대로 돌려줌.
"""
from __future__ import annotations

import datetime
import os
import subprocess
import sys
from pathlib import Path


def main():
    log = Path(sys.argv[1])
    cmd = sys.argv[2:]
    if cmd and cmd[0].endswith(".py"):
        cmd = [sys.executable, "-u"] + cmd
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
        for raw in p.stdout:
            line = raw.decode("utf-8", errors="replace")
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        code = p.wait()
        f.write(f"===== 종료 코드 {code}\n")
    sys.exit(code)


if __name__ == "__main__":
    main()
