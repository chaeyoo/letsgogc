# RAPV-Assistant — 배포용 컨테이너 이미지
# 컨테이너 호스트(Railway·Fly.io·Render(Docker)·Cloud Run 등) 어디에나 올라간다.
# 의존성이 순수 파이썬(numpy/torch 없음)이라 이미지가 가볍고 콜드스타트가 빠르다.
FROM python:3.11-slim

WORKDIR /app

# 런타임 의존성만 설치(테스트 의존성 requirements-dev 는 제외)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 복사
COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# 기동 게이트: 배포 전 점검(preflight) 통과 후에만 서버 시작(run.sh 와 동일 규율 —
# 데이터·설정·안전장치 자가 테스트가 실패하면 컨테이너가 뜨지 않는다). 플랫폼이
# 주입하는 $PORT 로 0.0.0.0 바인딩(없으면 8000). 오프라인 모드가 기본이며, LLM
# 모드는 ANTHROPIC_API_KEY 환경변수를 줄 때만 활성화된다.
CMD ["sh", "-c", "python -m src.preflight && exec uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
