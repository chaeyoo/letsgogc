#!/usr/bin/env bash
# RA-Assistant 원커맨드 실행 스크립트
set -e

cd "$(dirname "$0")"

# 1) 가상환경 준비
if [ ! -d ".venv" ]; then
  echo "▶ 가상환경 생성 및 의존성 설치..."
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -r requirements.txt
fi

# 2) .env 로드 (있으면)
if [ -f ".env" ]; then
  set -a; . ./.env; set +a
fi

# 3) 서버 실행
echo "▶ RA-Assistant 서버 시작 → http://127.0.0.1:8000"
exec .venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
