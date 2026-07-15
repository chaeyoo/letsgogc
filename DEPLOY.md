# 배포 가이드 — RAPV-Assistant

이 데모(`project/`)를 **공개 URL**로 올리는 방법. 면접관·리크루터가 링크를 눌러
바로 써볼 수 있게 하는 것이 목적이다.

## 0. 먼저 알아둘 것

- **오프라인 모드로 배포한다(권장).** API 키 없이도 RAG·PV·검증 전 기능이 동작한다.
  공개 URL 에 `ANTHROPIC_API_KEY` 를 넣으면 **방문자가 당신의 API 비용을 태우므로**
  넣지 않는다. LLM 모드는 면접 라이브 시연처럼 통제된 상황에서만 임시로 켠다(§4).
- 이 앱은 **오래 사는 서버**다(부팅 시 1회 RAG 인덱싱 + 인메모리 계기판). 그래서
  서버리스(Vercel)보다 **영속 프로세스/컨테이너 호스트(Render·Railway·Fly)** 가 맞다.
- 리포에 배포 파일이 들어 있다: `render.yaml`(레포 루트), `project/Dockerfile`,
  `project/.dockerignore`.
- 자작 규제 코퍼스라 실 PII·비밀이 없어 공개 배포는 안전하다.

로컬에서 배포와 동일한 방식으로 미리 확인하려면:

```bash
cd project
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
PORT=8000 sh -c 'python -m src.preflight && \
  .venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port $PORT'
# → http://127.0.0.1:8000 접속, /health·/api/deadlines·/chat 확인
```

---

## 1. 경로 A — Render (권장, 가장 쉬움)

무료 티어. `render.yaml` 을 자동 인식한다.

1. 코드를 GitHub 에 푸시(이미 되어 있으면 생략).
2. [dashboard.render.com](https://dashboard.render.com) 로그인 → **New +** → **Blueprint**.
3. 이 GitHub 리포를 선택 → Render 가 레포 루트의 `render.yaml` 을 읽어 **rapv-assistant**
   웹 서비스를 자동 구성한다(rootDir=`project`, 빌드=의존성 설치, 기동=preflight+uvicorn).
4. **Apply** → 첫 빌드·배포가 돈다(2~3분). 로그에 `배포 전 점검 … 통과` 후 uvicorn
   기동이 보이면 정상.
5. 발급된 URL(`https://rapv-assistant-xxxx.onrender.com`)에 접속 → 챗 UI 가 뜬다.

> 무료 티어는 15분 유휴 시 슬립한다 — **첫 접속만 콜드스타트 ~30초**, 이후 빠름.
> 면접 직전에 한 번 열어 깨워두면 매끄럽다.

블루프린트 대신 수동으로 만들려면: New + → **Web Service** → 리포 연결 →
Root Directory `project` · Build `pip install -r requirements.txt` ·
Start `python -m src.preflight && uvicorn src.api.main:app --host 0.0.0.0 --port $PORT` ·
Health Check Path `/health`.

---

## 2. 경로 B — Railway / Fly.io (Docker, 슬립 없음)

`project/Dockerfile` 을 그대로 쓴다.

**Railway**: [railway.app](https://railway.app) → New Project → Deploy from GitHub →
리포 선택 → Settings 에서 **Root Directory = `project`** 지정(Dockerfile 자동 감지) →
Deploy. 포트는 Railway 가 `$PORT` 로 주입(Dockerfile 이 처리).

**Fly.io**: `cd project && fly launch`(Dockerfile 감지, `fly.toml` 생성) → `fly deploy`.
`internal_port` 를 8080 등으로 두면 Dockerfile 의 `${PORT:-8000}` 대신 `PORT` 를
환경변수로 넘겨 맞춘다.

---

## 3. 배포 확인

URL 뒤에 붙여 확인:

- `/` — 챗 UI(질문 입력 → 근거·출처·검증 배지 표시)
- `/health` — `mode: offline`, 인덱스 문서/청크 수, 검증 게이트 계기판 JSON
- `/api/deadlines` — 마감일 목록 JSON
- `/chat` (POST) — `{"message":"..."}` 로 답변·citations·verification 반환

예: `curl -s https://<your-url>/health`

---

## 4. (선택) LLM 모드 켜기 — 통제된 시연에만

오프라인 모드는 규칙 라우터라 도구를 하나씩 부른다. LLM 모드는 Claude 가 복합 질문
("GMP 변경인데 뭘 준비하고 언제까지?")에 여러 도구를 **스스로 연쇄 호출**해 자연어로
종합한다 — 시각적으로 가장 인상적인 부분이다.

- **공개 URL 에는 켜지 말 것**(방문자가 API 비용 소모). 면접 시연이면 시연 직전에
  플랫폼 대시보드에서 `ANTHROPIC_API_KEY` 환경변수를 추가 → 재배포 → 시연 후 제거.
- 또는 **로컬**에서만 켠다: `project/.env` 에 `ANTHROPIC_API_KEY=sk-ant-...` 후 `./run.sh`.
- 모델 기본값은 `claude-opus-4-8`(`LLM_MODEL` 로 변경 가능).
- 키 오류·네트워크 실패는 조용한 폴백 없이 **명시적 안내**로 처리되므로 안전하다.

---

## 5. 트러블슈팅

| 증상 | 원인·해결 |
|---|---|
| 빌드는 됐는데 기동 실패 | 로그에 preflight 실패가 보이면 데이터·설정 결함 — 로컬에서 `python -m src.preflight` 로 재현·수정 후 재배포(이것이 fail-closed 게이트의 의도된 동작) |
| 404 / 앱이 안 뜸 | Root Directory 가 `project` 인지 확인(레포 루트가 아님) |
| 첫 접속이 느림 | 무료 티어 콜드스타트(~30초, 정상) — 유료 플랜이면 슬립 없음 |
| 포트 에러 | 기동 명령이 `--port $PORT` 인지 확인(플랫폼이 포트를 주입) |
| LLM 모드가 안 켜짐 | `ANTHROPIC_API_KEY` 환경변수 설정 후 **재배포** 필요(빌드 시점 주입) |

---

**요약**: Render 에 `render.yaml` 블루프린트로 오프라인 모드 배포 → 공개 URL 획득.
LLM 라이브 시연이 필요할 때만 키를 임시로 켠다.
