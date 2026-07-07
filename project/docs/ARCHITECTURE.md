# 아키텍처 & 설계 결정 (Design Notes)

이 문서는 면접에서 "왜 이렇게 만들었나"를 설명하기 위한 설계 근거를 정리한다.

---

## 1. 전체 데이터 흐름

```
[부팅 시 1회]
 규제문서(.md) ─loader─▶ Document ─chunker─▶ Chunk[] ─embedder.fit─▶ TF-IDF IDF 학습
                                                     └─embed─▶ 벡터 인덱스(+BM25 인덱스)

[질의 시]
 사용자 질문 ─▶ FastAPI /chat ─▶ RaAgent.chat()
     ├─(LLM 모드)  Claude ⇄ MCP 도구 tool-use 루프 (관찰→생각→행동 반복)
     └─(오프라인)  규칙 라우터 ─▶ MCP 도구 1회 호출 ─▶ grounded 답변 조립
                                   │
                     search_regulations ─▶ RAG(하이브리드 검색→리랭킹) ─▶ 근거+출처
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
2. **2차 (리랭킹):** `(질의 토큰 커버리지 + 정확 구문 매칭 + 섹션제목 가중)`으로
   재점수해 상위 N(=3)만 남긴다. Cross-Encoder 리랭커의 경량 근사.

> **Bi-Encoder vs Cross-Encoder** 개념을 코드로 표현: 1차는 빠른 벡터 유사도(넓게),
> 2차는 질의-청크를 함께 보는 정밀 재정렬(좁게). 실무에선 2차를 실제 Cross-Encoder/LLM 리랭커로 교체.

## 3. MCP 서버 (`mcp_server/server.py`)

- **왜 MCP인가:** 도구와 모델을 표준 규격으로 분리하면 N×M 통합이 N+M으로 준다.
  GC `Hey.GC 2.0`이 사내 시스템을 MCP로 통합하는 것과 같은 이유.
- **노출 primitive:**
  - Tools: `search_regulations`, `get_ra_deadlines`, `get_submission_checklist`, `list_regulation_documents`
  - Resource: `regulation://{doc_id}` (문서 원문)
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
| **출처 추적** | 모든 검색 답변에 문서·섹션 citation 부착 |
| **환각 억제** | "근거 없으면 모른다" 시스템 프롬프트 + grounded 검색 강제 |
| **가용성** | LLM 키 없이도 폴백 동작(무중단 데모) |
| **확장성** | MCP로 도구 분리 → 새 사내 시스템을 도구로 추가만 하면 됨 |
| **관측성(확장 지점)** | 도구 호출 트레이스를 응답에 포함(실무에선 LangSmith 등으로 확장) |

## 6. 의도적으로 뺀 것 (범위 관리, MVP 원칙)

- 실제 Vector DB(pgvector 등) → 인메모리 스토어로 대체(개념 동일, 배포 단순).
- 상용 임베딩/리랭커 모델 → 순수 파이썬 근사(무거운 의존성 제거, 어디서든 실행).
- 인증·영속 DB·CI → MVP 범위 밖. 확장 지점만 코드에 표시.

> 원칙: "완벽함보다 **작동+설명**". 12일 안에 GC 방향(Agentic+MCP)과 정확히 겹치는
> 데모를 **끝까지 작동**시키는 데 집중했다.
