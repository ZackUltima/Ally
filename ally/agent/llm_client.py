"""LLMClient interface: dialogue_turn(), classify_reply() -> ReplyLabel,
classify_sentiment() -> SpeechSentiment. Text only — no image input (ADR-007/015); local VLM captions are
never passed in (ADR-018).

The only module in the codebase that imports `anthropic`. Structured outputs for ReplyLabel;
tool schemas declared strict; the persona prompt is prompt-cached. Model IDs come from config, never here.

Sprint: Weeks 4–5 (B-9, talking prototype), completed in S4. See docs/context/architecture.md "LLM client".
"""
