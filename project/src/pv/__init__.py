"""PV(Pharmacovigilance·약물감시) 도메인 로직.

이상사례(AE) 트리아지(중대성 판정·보고기한 계산)와
개인정보 비식별화(redaction) 등 PV 업무에 특화된 순수 로직 모듈.
MCP 도구(src/mcp_server)가 이 로직을 감싸 에이전트에 노출한다.
"""
