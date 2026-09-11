"""JSON protocol for agent communication."""

from .json_schema import (
    MessageType,
    StepMessage,
    ProposeSkillMessage,
    StopMessage,
    ToolCall,
    parse_message,
    validate_message,
)
from .repair import extract_json, repair_json, repair_truncated_json
from .version import (
    INTERACTION_PROTOCOL_ENV,
    effective_interaction_protocol,
    interaction_protocol_v2_enabled,
)

__all__ = [
    "MessageType",
    "StepMessage",
    "ProposeSkillMessage",
    "StopMessage",
    "ToolCall",
    "parse_message",
    "validate_message",
    "extract_json",
    "repair_json",
    "repair_truncated_json",
    "INTERACTION_PROTOCOL_ENV",
    "effective_interaction_protocol",
    "interaction_protocol_v2_enabled",
]
