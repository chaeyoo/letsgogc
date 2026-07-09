# 아키텍처 & 설계 결정 (Design Notes)

이 문서는 면접에서 "왜 이렇게 만들었나"를 설명하기 위한 설계 근거를 정리한다.

---

## 1. 전체 데이터 흐름

```
[부팅 시 1회]
 규제문서(.md) ─loader─▶ Document ─chunker─▶ Chunk[] ─embedder.fit─▶ TF-IDF IDF 학습
                                                     └─embed─▶ 벡터 인덱스(+BM25 인덱스)

[질의 시]
 사용자 질문 ─▶ FastAPI /chat ─▶ RaAgent.chat() ─▶ 🔒 PII 마스킹(입구, 이후 전 경로 안전)
     ├─(LLM 모드)  Claude ⇄ MCP 도구 tool-use 루프 (관찰→생각→행동 반복)
     └─(오프라인)  규칙 라우터 ─▶ MCP 도구 1회 호출 ─▶ grounded 답변 조립
                                   │
                     search_regulations ─▶ RAG(질의확장→하이브리드 검색→리랭킹) ─▶ 근거+출처
                     assess_adverse_event ─▶ PV 트리아지+인과성(WHO-UMC) 제안+용어 코딩 ─▶ 근거 규정 부착
                     draft_ae_report ─▶ 최소보고요건(ICH E2D) 검증 ─▶ ICSR(KAERS) 초안 조립
                     get_ra_deadlines / get_submission_checklist ─▶ 업무 데이터
```

## 2. RAG 파이프라인 세부 (검색 '최적화'가 드러나는 곳)

### 2.1 청킹 (`rag/chunker.py`)
- **구조 우선 분할:** markdown 헤딩(`##`) 단위로 먼저 쪼개 문맥(섹션)을 보존한다.
- **크기 제한 + overlap:** 긴 섹션은 `CHUNK_SIZE=500`자, `CHUNK_OVERLAP=80`자로
  슬라이딩 분할해 경계에서 정보가 잘리는 것을 막는다.
- **경량 Contextual Retrieval:** 각 청크 앞에 `[문서제목 > 섹션]`을 붙여
  청크가 고립돼도 어느 맥락인지 검색·표시에 활용한다.

### 2.2 임베딩 (`rag/embedder.py`)
- 오프라인 기본값은 **TF-IDF 희소 벡터**(순수 파이썬, 무거운 의존성 0).
- `EmbeddingProvider` 프로토콜로 **교체 가능** — 실무에선 이 자리에 상용 임베딩 API를 끼운다.
- 한국어 매칭 보강: `textutil.tokenize()`가 어절 + **CJK 문자 bi-gram**을 함께 생성해
  조사·띄어쓰기 변형("감기약"↔"감기")에 대응한다(경량 형태소 근사).

### 2.3 하이브리드 검색 + 리랭킹 (`rag/retriever.py`)
2단계 구조로 **재현율(넓게)과 정밀도(좁게)를 모두** 잡는다.
1. **1차 (하이브리드):** 벡터(TF-IDF 코사인) + 키워드(**BM25**) 점수를 min-max 정규화 후
   `HYBRID_ALPHA`로 가중 결합해 top_k(=8) 회수. → 의미 유사어와 고유명사·코드 모두 커버.
2. **2차 (리랭킹):** 4신호 가중합 `(본문 커버리지(idf^p 가중) + 정확 구문 매칭 +
   섹션제목 매칭 + 문서제목 정합)`으로 재점수해 상위 N(=3)만 남긴다.
   Cross-Encoder 리랭커의 경량 근사. 섹션 신호와 제목 신호를 분리한 것은
   BM25F처럼 필드별 매칭을 따로 보는 설계 — 같은 문서 안에서 '정답 섹션'을
   고르는 신호(섹션)와 하드네거티브 문서를 거르는 반증 신호(제목)의 역할이 다르다.

> **Bi-Encoder vs Cross-Encoder** 개념을 코드로 표현: 1차는 빠른 벡터 유사도(넓게),
> 2차는 질의-청크를 함께 보는 정밀 재정렬(좁게). 실무에선 2차를 실제 Cross-Encoder/LLM 리랭커로 교체.

### 2.4 질의 확장 (`rag/synonyms.py`)
- 사용자 구어("부작용"·"설명서")와 문서 정식 용어("이상사례"·"첨부문서")의
  **어휘 불일치**를 도메인 동의어 사전으로 메운다(단방향: 구어→정식 용어).
- **1단계 회수에 전 가중으로 적용**하고, 2단계 리랭킹은 원 질의 토큰 기준으로
  재점수하되 **확장 토큰은 절반 가중의 보조 신호**로만 반영 — 확장어가 정밀도
  신호를 희석하지 않으면서도, 원 질의 토큰이 문서에 전혀 없는 구어 질의에서
  리랭커가 판별력을 잃지 않게 한다. Hit@1 0.867→0.967(eval 검증).
- LLM 질의 재작성 대비 결정론적(감사 가능)·저지연·무비용. 사전 항목은 eval로
  검증해 채택/제외한다(예: "DMF" 확장은 주제 표류를 일으켜 제외, "심각→중대한"·
  "섞이→교차오염"은 오류 분석에서 발굴해 추가).

## 3. MCP 서버 (`mcp_server/server.py`)

- **왜 MCP인가:** 도구와 모델을 표준 규격으로 분리하면 N×M 통합이 N+M으로 준다.
  GC `Hey.GC 2.0`이 사내 시스템을 MCP로 통합하는 것과 같은 이유.
- **노출 primitive (3종 완성):**
  - Tools: `search_regulations`, `assess_adverse_event`(PV 트리아지+인과성+코딩),
    `draft_ae_report`(ICSR 초안+최소요건 검증), `get_ra_deadlines`,
    `get_submission_checklist`, `list_regulation_documents`
  - Resource: `regulation://{doc_id}` (문서 원문)
  - Prompt: `pv_case_intake` (케이스 처리 SOP — 어느 클라이언트가 붙어도 같은 절차)
- **결정론적 도구 원칙:** 보고기한 계산 같은 컴플라이언스 판정은 LLM이 아니라
  규칙 기반 도구(`src/pv/triage.py`)가 수행한다 — LLM은 도구 선택·설명만 담당.
- **도구는 의도 단위로 분리:** "언제까지 보고?"(판정)와 "보고서 만들어줘"(산출물)는
  다른 도구(`assess` vs `draft`)로 — 단, 같은 판정 규칙 모듈을 공유해 결과 불일치가 없다.
- **도구 설명이 곧 LLM의 사용설명서:** 각 도구의 docstring·타입힌트가 에이전트의
  도구 선택 정확도를 좌우한다 → FDE의 실력이 드러나는 지점.
- **두 실행 모드:** stdio(독립 실행, Claude Desktop/Cursor 연결) / 인메모리(`Client(mcp)`, 데모 기본).

## 4. 에이전트 (`agent/agent.py`)

- **LLM 모드:** Anthropic `messages.create(tools=...)` tool-use 루프. `stop_reason == "tool_use"`인 동안
  MCP로 도구를 실행하고 결과를 되먹여 최종 답을 만든다. 최대 6스텝(무한루프 방지).
- **오프라인 모드:** 규칙 기반 인텐트 라우팅(마감일/체크리스트/검색) 후 grounded 답변.
  API 키 없이도 데모가 항상 동작하도록 하는 **graceful degradation**.
- **공통:** 도구는 반드시 MCP를 통해 호출 → 모델/도구 분리 원칙 유지.

## 5. 신뢰성 설계 (제약 규제 산업 감각)

| 요구 | 구현 |
|---|---|
| **출처·버전 추적** | 모든 검색 답변에 문서·섹션·**버전·시행일** citation 부착 |
| **환각 억제** | 근거 관련도+커버리지 두 신호로 **abstention**(근거 없으면 "모른다") · grounded 검색 강제 |
| **버전 안전성** | 폐지(superseded) 구판 자동 제외 · `as_of`로 과거 시점 규정 조회 |
| **개인정보 보호** | 에이전트 입구에서 **PII 마스킹**(`pv/redactor.py`) — 외부 LLM API·로그·트레이스에 원문 비유출, 응답엔 유형·건수만 |
| **컴플라이언스 계산 분리** | 보고기한·중대성·최소보고요건 판정은 **규칙 기반 도구**(`pv/triage.py`·`pv/report.py`) — LLM 추론에 맡기지 않아 감사 가능 |
| **판단 확신 수준 구분** | 닫힌 목록 '대조'(중대성)는 판정으로, 종합 '판단'(인과성)은 **제안+follow-up 질문**으로(`pv/causality.py`) |
| **가용성** | LLM 키 없이도 폴백 동작(무중단 데모) |
| **견고성** | MCP 도구 실패를 크래시 대신 흡수해 모델에 되먹임(자가 복구) |
| **관측성** | 스텝별 지연·성패를 span 트레이스+구조화 로그로 기록(`observability.py`), 응답에 `trace`·`latency_ms` 노출 |
| **검증성** | pytest 71케이스 + eval(검색·신뢰성·**PV 워크플로**)을 CI에서 매 푸시 실행 |
| **확장성** | MCP로 도구 분리 · `EmbeddingProvider`로 임베더 교체(TF-IDF/해싱/Voyage) |

### 5.1 답변 신뢰성 측정 (`eval/faithfulness.py`)
검색 정확도(evaluate)와 별개로 **생성 안전성**을 잰다:
- **AnswerGroundedness**: 답이 근거로 뒷받침되는가(1.000)
- **AbstentionAccuracy**: 범위 밖 질문에 지어내지 않고 회피하는가(1.000) · OverAbstain 0.000
- abstention은 `SCORE_FLOOR`·`COVERAGE_FLOOR` 두 신호가 **둘 다** 약할 때만 발동(AND) → 과회피 방지.
- 커버리지 신호는 **확장 질의 기준**으로 계산한다 — 검색은 동의어 확장으로 정답을
  찾는데 회피 판정만 원 질의로 보면 '정답을 찾고도 모른다고 답하는' 자기모순이 생긴다.
- 문턱값은 범위내/범위밖 점수 분포를 실측해 마진 중앙으로 보정(리랭커가 바뀌면 재보정).

### 5.2 버전 인지 검색 (`retriever._is_active`)
규제문서 frontmatter의 `version`·`effective_date`·`status`를 청크로 전파하고,
세 검색 모드(벡터/하이브리드/리랭킹)가 **공통 후보 필터**를 거치게 해 폐지본을 일관되게 제외한다.
개정 이력이 필요한 경우 `include_superseded`로 구판까지 노출.

## 6. 의도적으로 뺀 것 (범위 관리, MVP 원칙)

- 실제 Vector DB(pgvector 등) → 인메모리 스토어로 대체(개념 동일, 배포 단순).
- 상용 임베딩/리랭커 모델 → 순수 파이썬 근사 + **실제 API 경로(VoyageEmbedder)는 코드로 표시**.
- 인증·영속 DB → MVP 범위 밖(확장 지점만 코드에 표시).
- CI는 **포함**했다(pytest+eval 회귀) — 확장이 아니라 신뢰성의 기본이라 판단.

> 설계 결정별 "왜 이렇게 했나 + 면접 예상질문 대응"은 [`docs/면접노트.md`](면접노트.md)에 정리했다.

> 원칙: "완벽함보다 **작동+설명**". 12일 안에 GC 방향(Agentic+MCP)과 정확히 겹치는
> 데모를 **끝까지 작동**시키는 데 집중했다.
