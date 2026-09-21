"""LLMClient interface: dialogue_turn(), verify_keyframe() -> VerifyResult, classify_reply() -> ReplyLabel.

The only module in the codebase that imports `anthropic`. Structured outputs for VerifyResult/ReplyLabel;
tool schemas declared strict; the persona prompt is prompt-cached. Model IDs come from config, never here.

Sprint: S3. See docs/context/architecture.md "LLM client".
"""
