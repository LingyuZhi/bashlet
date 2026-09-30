"""The action protocol: an experimental condition loaded from versioned config.

SFT demonstrations, student rollouts and evaluation all share one rendered
text. Any change needs a new protocol name and regenerated demonstrations.
"""

import hashlib
import json
import re
from dataclasses import asdict

from .config import ExperimentConfig, ProtocolConfig, load_config
from .models import Bash, InvalidAction, Observation, Submit

# Appended to the observation template when fingerprinting. It describes the
# rendering rule and is not itself shown to the agent.
_OBSERVATION_NOTE = " (stdout/stderr get a final newline if missing; not escaped)"
_BLOCK = re.compile(r"^```bash[ \t]*\n(.*?)\n?^```[ \t]*$", re.DOTALL | re.MULTILINE)
_DEFAULT = load_config()

PROTOCOL_VERSION = _DEFAULT.protocol.name
SYSTEM_PROMPT = _DEFAULT.protocol.system_prompt


def parse_action(text: str, truncated: bool) -> Bash | Submit | InvalidAction:
    if truncated:
        return InvalidAction("The reply hit the generation length limit")
    blocks = _BLOCK.findall(text)
    if len(blocks) != 1:
        return InvalidAction(f"Expected exactly one ```bash block, found {len(blocks)}")
    command = blocks[0]
    if command.strip() == "submit":
        return Submit()
    if not command.strip():
        return InvalidAction("The ```bash block is empty")
    return Bash(command)


def _line_terminated(text: str) -> str:
    return text if not text or text.endswith("\n") else text + "\n"


def render_observation(observation: Observation, template: str) -> str:
    """Tag contents are not escaped; the raw Observation is recorded beside it."""
    return template.format(
        status=observation.status,
        exit_code=observation.exit_code,
        stdout=_line_terminated(observation.stdout),
        stderr=_line_terminated(observation.stderr),
    )


def render_invalid(action: InvalidAction, template: str) -> str:
    return template.format(reason=action.reason)


def is_verification(action: Bash, feedback_unchanged: bool, pattern: str) -> bool:
    return feedback_unchanged and re.fullmatch(pattern, action.command) is not None


def definition(protocol: ProtocolConfig) -> dict[str, str]:
    """The canonical protocol record. Its hash is the protocol fingerprint."""
    return {
        "name": protocol.name,
        "system_prompt": protocol.system_prompt,
        "user_prompt": protocol.user_prompt,
        "observation_format": protocol.observation_template + _OBSERVATION_NOTE,
        "invalid_action_format": protocol.invalid_action_format,
        "history": protocol.history,
        "verification": (
            f"bash command fully matches {protocol.verification_pattern!r} while "
            "feedback_tests.py is unchanged; passed iff status ok"
        ),
    }


def fingerprint(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def fingerprints(config: ExperimentConfig) -> dict[str, str]:
    """Protocol, environment, and budget only. The model block is not included."""
    return {
        "protocol": fingerprint(definition(config.protocol)),
        "environment": fingerprint(asdict(config.environment)),
        "budget": fingerprint(asdict(config.budget)),
    }


def describe(protocol: ProtocolConfig) -> dict[str, str]:
    """The protocol as recorded in every run, with its fingerprint."""
    recorded = definition(protocol)
    return recorded | {"sha256": fingerprint(recorded)}


PROTOCOL_SHA256 = fingerprint(definition(_DEFAULT.protocol))
