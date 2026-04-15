"""
app_profile.py — Config-Driven Application Profile & Guardrails

Loads a YAML profile from `profiles/{APP_PROFILE}.yaml` to define:
  - Application name, tagline, and welcome message
  - Domain-specific system prompts and personality
  - Guardrail rules (allowed/blocked topics, strictness)
  - File handling labels and restrictions

Usage:
    from app_profile import profile

    # Access profile fields
    profile.app_name          # "ClaimAssist AI"
    profile.system_prompt     # Full system prompt string
    profile.welcome_message   # Welcome message for chat start

    # Check guardrails
    is_allowed, message = profile.check_guardrails("tell me a joke")
    # → (False, "🚫 I'm ClaimAssist AI, ...")
"""

import logging
import os
import re
from pathlib import Path
from typing import Tuple

import yaml

logger = logging.getLogger(__name__)

# ── Profile directory ─────────────────────────────────────────────
PROFILES_DIR = Path(__file__).parent / "profiles"


class AppProfile:
    """Typed wrapper around a YAML application profile."""

    def __init__(self, profile_name: str = "default"):
        self.profile_name = profile_name
        self._data = self._load_profile(profile_name)

        # ── Parsed sections ───────────────────────────────────────
        app_section = self._data.get("app", {})
        domain_section = self._data.get("domain", {})
        guardrails_section = self._data.get("guardrails", {})
        personality_section = self._data.get("personality", {})
        file_section = self._data.get("file_handling", {})

        # ── App identity ──────────────────────────────────────────
        self.app_name: str = app_section.get("name", "Genius AI")
        self.tagline: str = app_section.get("tagline", "Your AI Assistant")
        self.welcome_message: str = app_section.get("welcome_message", "").strip()

        # ── Domain ────────────────────────────────────────────────
        self.domain_description: str = domain_section.get("description", "")
        self.allowed_topics: list[str] = [
            t.lower() for t in domain_section.get("allowed_topics", [])
        ]
        self.blocked_topics: list[str] = [
            t.lower() for t in domain_section.get("blocked_topics", [])
        ]
        self.strictness: str = domain_section.get("strictness", "relaxed")

        # ── Guardrails ────────────────────────────────────────────
        self.off_topic_response: str = guardrails_section.get(
            "off_topic_response", ""
        ).strip()
        self.no_data_response: str = guardrails_section.get(
            "no_data_response", ""
        ).strip()

        # ── Personality ───────────────────────────────────────────
        self.system_prompt: str = personality_section.get(
            "system_prompt", ""
        ).strip()
        self.tone: str = personality_section.get("tone", "friendly")
        self.use_emojis: bool = personality_section.get("use_emojis", True)
        self.format_preference: str = personality_section.get(
            "format_preference", "bullet_points"
        )

        # ── File handling ─────────────────────────────────────────
        self.data_source_label: str = file_section.get(
            "data_source_label", "your uploaded data"
        )
        ext = file_section.get("allowed_extensions", {})
        self.allowed_structured_ext: set[str] = set(
            ext.get("structured", [".xlsx", ".xls", ".csv", ".json"])
        )
        self.allowed_unstructured_ext: set[str] = set(
            ext.get("unstructured", [".pdf", ".txt", ".md"])
        )

        logger.info(
            f"Loaded profile '{profile_name}': "
            f"app={self.app_name}, strictness={self.strictness}, "
            f"allowed_topics={len(self.allowed_topics)}, "
            f"blocked_topics={len(self.blocked_topics)}"
        )

    # ── Profile Loading ───────────────────────────────────────────

    @staticmethod
    def _load_profile(name: str) -> dict:
        """Load a YAML profile by name, falling back to default."""
        profile_path = PROFILES_DIR / f"{name}.yaml"

        if not profile_path.exists():
            logger.warning(
                f"Profile '{name}' not found at {profile_path}. "
                f"Falling back to 'default'."
            )
            profile_path = PROFILES_DIR / "default.yaml"

        if not profile_path.exists():
            logger.error("No profile files found! Using empty config.")
            return {}

        with open(profile_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        logger.info(f"Loaded profile from: {profile_path}")
        return data or {}

    # ── Guardrail Check ───────────────────────────────────────────

    def check_guardrails(
        self, query: str, has_data: bool = False
    ) -> Tuple[bool, str]:
        """
        Check if a query is allowed by the profile's guardrails.

        Logic (fast keyword matching — no LLM call):
          1. strictness == "relaxed" → always allow
          2. Check blocked_topics → instant reject
          3. has_data == True → allow (user uploaded data, let them query it)
          4. Check allowed_topics → pass if any keyword matches
          5. strictness == "strict" → reject (no match = off-topic)
          6. strictness == "moderate" → allow (benefit of the doubt)

        Returns:
            (is_allowed: bool, rejection_message: str)
            If is_allowed is True, rejection_message is empty.
        """
        # 1. Relaxed mode — no guardrails
        if self.strictness == "relaxed":
            return True, ""

        query_lower = query.lower().strip()

        # 2. Check blocked topics — instant reject
        for blocked in self.blocked_topics:
            if blocked in query_lower:
                logger.info(
                    f"Guardrail BLOCKED: query matched blocked topic '{blocked}'"
                )
                return False, self.off_topic_response

        # 3. If user has data loaded, allow data queries
        if has_data:
            return True, ""

        # 4. Check allowed topics — pass if any match
        if self.allowed_topics:
            for topic in self.allowed_topics:
                if topic in query_lower:
                    return True, ""

            # No allowed topic matched
            if self.strictness == "strict":
                logger.info(
                    f"Guardrail BLOCKED (strict): no allowed topic match"
                )
                return False, self.off_topic_response

        # 5. Greetings and simple messages always pass
        greeting_patterns = [
            r"^(hi|hello|hey|good morning|good evening|good afternoon)",
            r"^(thanks|thank you|ok|okay|sure|yes|no|bye|goodbye)",
        ]
        for pattern in greeting_patterns:
            if re.match(pattern, query_lower):
                return True, ""

        # 6. Moderate mode — benefit of the doubt
        if self.strictness == "moderate":
            return True, ""

        # Default: allow
        return True, ""

    # ── File Extension Check ──────────────────────────────────────

    def is_file_allowed(self, filename: str) -> Tuple[bool, str]:
        """Check if a file extension is allowed by this profile."""
        ext = Path(filename).suffix.lower()
        all_allowed = self.allowed_structured_ext | self.allowed_unstructured_ext

        if ext in all_allowed:
            return True, ""

        allowed_list = ", ".join(sorted(all_allowed))
        return False, (
            f"⚠️ **{self.app_name}** doesn't support `{ext}` files.\n"
            f"Allowed formats: {allowed_list}"
        )

    # ── Convenience ───────────────────────────────────────────────

    def get_system_prompt_with_summary(self, summary: str = "") -> str:
        """Build full system prompt with optional conversation summary."""
        prompt = self.system_prompt
        if summary:
            prompt += f"\n\n## Previous Conversation Summary\n{summary}"
        return prompt

    def __repr__(self) -> str:
        return (
            f"AppProfile(name='{self.profile_name}', "
            f"app='{self.app_name}', strictness='{self.strictness}')"
        )


# ══════════════════════════════════════════════════════════════════
# SINGLETON — loaded once on import
# ══════════════════════════════════════════════════════════════════

_profile_name = os.getenv("APP_PROFILE", "default")
profile = AppProfile(_profile_name)
