from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Literal, get_args

from sqlalchemy.orm import Session, sessionmaker

from llm.reasoning.engine import Budget
from llm.studio.access_policy import AccessPolicy

ProfileId = Literal["lite", "medium", "high", "very_high"]
PROFILE_IDS: tuple[str, ...] = get_args(ProfileId)
ALLOWED_KEY = "profiles.allowed"
MIN_PROMPT_TOKENS = 32
CONTEXT_MARGIN = 16


class ProfilePolicyError(ValueError):
    pass


@dataclass(frozen=True)
class ReasoningConfig:
    enabled: bool
    max_candidates: int
    verification_attempts: int


@dataclass(frozen=True)
class ResourceLimits:
    max_total_new_tokens: int
    max_concurrent_generations: int


@dataclass(frozen=True)
class InferenceProfile:
    id: ProfileId
    max_output_tokens: int
    context_budget: int
    generation_timeout_seconds: float
    tool_budget_seconds: float
    reasoning: ReasoningConfig
    limits: ResourceLimits

    def budget(self, per_candidate: int) -> Budget:
        return Budget(
            max_candidates=self.reasoning.max_candidates,
            max_retries=self.reasoning.verification_attempts,
            max_new_tokens_total=min(
                self.limits.max_total_new_tokens, per_candidate * self.reasoning.max_candidates
            ),
            max_seconds=self.tool_budget_seconds,
            max_new_tokens_per_candidate=per_candidate,
        )


PROFILES: dict[str, InferenceProfile] = {
    "lite": InferenceProfile(
        "lite", 128, 256, 30.0, 0.0, ReasoningConfig(False, 1, 0), ResourceLimits(128, 1)
    ),
    "medium": InferenceProfile(
        "medium", 256, 512, 60.0, 0.0, ReasoningConfig(False, 1, 0), ResourceLimits(256, 1)
    ),
    "high": InferenceProfile(
        "high", 256, 1024, 90.0, 30.0, ReasoningConfig(True, 2, 1), ResourceLimits(768, 1)
    ),
    "very_high": InferenceProfile(
        "very_high", 512, 2048, 180.0, 60.0, ReasoningConfig(True, 4, 2), ResourceLimits(2048, 1)
    ),
}


def unavailable_reason(profile: InferenceProfile, context_length: int | None) -> str | None:
    if context_length is None:
        return "model_not_loaded"
    if profile.max_output_tokens + MIN_PROMPT_TOKENS + CONTEXT_MARGIN > context_length:
        return "context_too_small"
    return None


def profile_view(
    profile: InferenceProfile, context_length: int | None, allowed: set[str] | None
) -> dict[str, Any]:
    reason = unavailable_reason(profile, context_length)
    return {
        **asdict(profile),
        "available": reason is None,
        "unavailable_reason": reason,
        "allowed": allowed is None or profile.id in allowed,
    }


def effective_output_tokens(profile: InferenceProfile, context_length: int) -> int:
    return max(
        1, min(profile.max_output_tokens, context_length - CONTEXT_MARGIN - MIN_PROMPT_TOKENS)
    )


class ProfileAccessPolicy(AccessPolicy):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        super().__init__(session_factory, ALLOWED_KEY)

    def update(self, identifiers: list[str]) -> list[str]:
        unknown = sorted(set(identifiers) - set(PROFILE_IDS))
        if unknown:
            raise ProfilePolicyError(json.dumps({"unknown": unknown}))
        return self.store(set(identifiers))
