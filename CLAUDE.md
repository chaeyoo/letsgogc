# CLAUDE.md

이 저장소는 GC녹십자 FDE(Forward Deployed Engineer) 지원 준비 자료다.
`project/` 에는 포트폴리오 데모 **RA-Assistant**, 루트와 `직접작성/` 에는 지원 문서(자소서·경력기술서 등)가 있다.

## RA·PV 표기 규칙 (필수)

RA-Assistant 는 **RA(인허가/규제업무)와 PV(약물감시) 두 도메인을 함께 다루는 도구**다.
과거에 RA 전용 → PV 중심으로 프레이밍이 오락가락한 이력이 있어, 전 문서를 RA·PV 로 통일했다(PR #27~#30).
**새 문서를 만들거나 기존 문서에 내용을 추가할 때도 이 통일을 유지해야 한다.**

### 반드시 "RA·PV" 로 쓰는 곳 — 프로젝트/도구 전체를 지칭할 때

- 프로젝트 소개·개요 문구: "제약 규제업무(RA·PV)용 …", "RA·PV 담당자를 위한 …"
- 사용자·에이전트 라벨(문서 본문, mermaid/SVG 다이어그램 포함): "RA·PV 담당자", "RA·PV 에이전트"
- MCP 도구 서버·업무 시스템 지칭: "RA·PV 도구 서버", "RA·PV 업무 시스템"
- `data/ra_tasks.json` 을 가리키는 레이블: "RA·PV 업무 데이터" (PV팀 마감(PSUR·신속보고)이 포함되어 있음)
- 채용공고 "제약/바이오 산업 이해" 매핑: "RA·PV 도메인 도구로 증명"
- 코드 docstring 에서 어시스턴트/서버 전체를 지칭할 때 (`src/agent/`, `src/api/`, `src/mcp_server/`)

구분 기호는 가운뎃점 **`RA·PV`** 로 통일한다 (`RA/PV`, `RA, PV` 지양).
문서 구조에서는 두 도메인을 대칭으로 다룬다 — RA 도메인(규제문서 검색 RAG + 마감일·체크리스트 도구)과
PV 도메인(`src/pv/` 케이스 처리 워크플로)이 항상 나란히 등장해야 하며, 한쪽만 도메인으로 소개하지 않는다.

### RA 또는 PV 단독 표기가 맞는 곳 — 바꾸지 말 것

- **제품명 "RA-Assistant"** — 고유명사이므로 그대로 둔다 (FastMCP 서버명, FastAPI title 포함)
- RA/PV **용어 자체의 정의·설명** (예: `description/dictionary.md` 의 RA·PV 항목)
- **한쪽 도메인 내부를 설명하는 문맥** (예: "RA 도메인" 장 안에서 RA 업무를 설명, `src/pv/` PV 워크플로 설명)
- `data/regulations/` **샘플 규제문서 코퍼스** (예: "RA 담당자 체크포인트" 섹션) — 검색 평가(Hit@1 등)의
  대상 데이터이므로 내용을 바꾸면 평가 수치가 흔들린다
- `직접작성/경력기술서_2.md` 의 **AS-IS 아카이브 섹션** — 구버전 보관용이므로 수정하지 않는다
- 파일명·식별자 (`ra_tasks.json`, `get_ra_deadlines` 등) — 코드 호환성 유지

### 이미지 파일 주의

`project/docs/architecture.svg` 를 수정하면 래스터본 **`architecture.png` 도 재렌더링해서 교체**해야 한다
(README·프로젝트_소개서에 삽입됨). 텍스트 검색으로는 PNG 속 구버전 라벨이 잡히지 않는다.

## 검증

`project/` 의 문서가 아닌 코드·데이터(`src/`, `data/`)를 수정했다면 커밋 전에 실행:

```bash
cd project
.venv/bin/python -m pytest -q      # 202케이스
.venv/bin/python -m src.preflight  # 배포 전 점검
```
