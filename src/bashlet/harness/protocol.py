"""The action protocol: an experimental condition loaded from versioned config.

Student rollouts and evaluation share the same rendered protocol. Changes
are recorded as new experimental conditions.
"""

import hashlib
import json
import re
from dataclasses import asdict

from bashlet.config import ExperimentConfig, ProtocolConfig, load_config
from bashlet.harness.models import Bash, InvalidAction, Observation

# Appended to the observation template when fingerprinting. It describes the
# rendering rule and is not itself shown to the agent.
_OBSERVATION_NOTE = " (merged output gets a final newline if missing; not escaped)"
_BLOCK = re.compile(r"^```bash[ \t]*\n(.*?)\n?^```[ \t]*$", re.DOTALL | re.MULTILINE)
_BLOCK_START = re.compile(r"^```bash[ \t]*$", re.MULTILINE)
_DEFAULT = load_config()

PROTOCOL_VERSION = _DEFAULT.protocol.name
SYSTEM_PROMPT = _DEFAULT.protocol.system_prompt


def parse_action(text: str, truncated: bool) -> Bash | InvalidAction:
    if truncated:
        return InvalidAction("The reply hit the generation length limit")
    blocks = list(_BLOCK.finditer(text))
    starts = len(_BLOCK_START.findall(text))
    if len(blocks) != 1 or starts != 1:
        return InvalidAction(f"Expected exactly one ```bash block, found {starts}")
    command = blocks[0].group(1)
    if not command.strip():
        return InvalidAction("The ```bash block is empty")
    return Bash(command)


def submission_text(observation: Observation, marker: str) -> str | None:
    if observation.status != "ok" or observation.exit_code != 0:
        return None
    first_line, separator, remainder = observation.output.partition("\n")
    if observation.output_truncated and not separator:
        return None
    return remainder if first_line == marker else None


def _line_terminated(text: str) -> str:
    return text if not text or text.endswith("\n") else text + "\n"


def render_observation(observation: Observation, protocol: ProtocolConfig) -> str:
    """Tag contents are not escaped; the raw Observation is recorded beside it."""
    output = observation.output
    head, tail = protocol.observation_head_chars, protocol.observation_tail_chars
    if len(output) > head + tail:
        elision = protocol.elision_template.format(
            omitted_chars=len(output) - head - tail
        )
        output = output[:head] + "\n" + elision + "\n" + output[-tail:]
    template = (
        protocol.timeout_template
        if observation.status == "timeout"
        else protocol.observation_template
    )
    feedback = template.format(
        exit_code=observation.exit_code,
        output=_line_terminated(output),
    )
    if observation.output_truncated:
        feedback += "\n" + protocol.output_limit_message
    return feedback


def render_invalid(action: InvalidAction, template: str) -> str:
    return template.format(reason=action.reason)


def definition(protocol: ProtocolConfig) -> dict[str, str | int]:
    """The canonical protocol record. Its hash is the protocol fingerprint."""
    return {
        "name": protocol.name,
        "action_format": (
            "exactly one non-empty, closed bash fenced block in the action body; "
            "a natural-language explanation before the block is optional; "
            "length-truncated replies are never executed"
        ),
        "thinking": (
            "when enabled, parse only the body after the last tokenizer thinking "
            "end marker; unfinished thinking is a format error; thinking blocks "
            "and submission markers never execute; raw thinking tokens remain "
            "in the append-only context, including a prompt-prefilled start marker"
        ),
        "system_prompt": protocol.system_prompt,
        "user_prompt": protocol.user_prompt,
        "observation_format": protocol.observation_template + _OBSERVATION_NOTE,
        "timeout_format": protocol.timeout_template + _OBSERVATION_NOTE,
        "observation_head_chars": protocol.observation_head_chars,
        "observation_tail_chars": protocol.observation_tail_chars,
        "elision_template": protocol.elision_template,
        "output_limit_message": protocol.output_limit_message,
        "invalid_action_format": protocol.invalid_action_format,
        "history": protocol.history,
        "submission": (
            f"first merged output line equals {protocol.submission_marker!r} "
            "and command completes with exit code 0; a byte-cutoff first line is "
            "not accepted; remaining output is submission text"
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


def describe(protocol: ProtocolConfig) -> dict[str, str | int]:
    """The protocol as recorded in every run, with its fingerprint."""
    recorded = definition(protocol)
    return recorded | {"sha256": fingerprint(recorded)}


PROTOCOL_SHA256 = fingerprint(definition(_DEFAULT.protocol))
