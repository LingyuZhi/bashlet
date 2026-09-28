"""The bash-block-v1 action protocol: an experimental condition, not a detail.

SFT demonstrations, student rollouts and evaluation all share this text. Any
change needs a new ``PROTOCOL_VERSION`` and regenerated demonstrations.
"""

import hashlib
import json
import re

from .models import Bash, InvalidAction, Observation, Submit

PROTOCOL_VERSION = "bash-block-v1"

SYSTEM_PROMPT = """\
You are a programming agent working in a Linux shell. The current directory \
contains task.md (the task) and feedback_tests.py (tests you may run). Write the \
implementation to solution.py.

Each reply must contain exactly one fenced block that starts with ```bash on its \
own line and ends with ``` on its own line. Text outside the block is allowed. \
The block is run with bash in the current directory, without network access. \
Each block runs in a fresh shell; only files in the current \
directory persist. To run the feedback tests, use a block that contains only \
`python3 feedback_tests.py`.

The result is returned as the next message:
<status>...</status>
<exit_code>...</exit_code>
<stdout>
...
</stdout>
<stderr>
...
</stderr>

When solution.py is final, reply with a block whose only content is submit:
```bash
submit
```
Submitting ends the task; solution.py is then graded with tests you cannot see. \
A reply without exactly one block, or cut off by the length limit, is not \
executed and counts against your step limit."""

# Analysis definition, not an agent-visible rule: a Bash step verifies the
# current candidate when it runs exactly the canonical feedback command on
# unmodified feedback tests, so its exit status is the tests' own verdict.
_VERIFICATION = re.compile(r"\s*python3 feedback_tests\.py\s*")
VERIFICATION_RULE = (
    f"bash command fully matches {_VERIFICATION.pattern!r} while feedback_tests.py "
    "is unchanged; passed iff status ok"
)

_BLOCK = re.compile(r"^```bash[ \t]*\n(.*?)\n?^```[ \t]*$", re.DOTALL | re.MULTILINE)


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


OBSERVATION_TEMPLATE = (
    "<status>{status}</status>\n"
    "<exit_code>{exit_code}</exit_code>\n"
    "<stdout>\n{stdout}</stdout>\n"
    "<stderr>\n{stderr}</stderr>"
)
INVALID_TEMPLATE = (
    "<invalid_action>{reason}</invalid_action>\n"
    "Nothing was executed. Reply with exactly one ```bash block."
)


def _line_terminated(text: str) -> str:
    return text if not text or text.endswith("\n") else text + "\n"


def render_observation(observation: Observation) -> str:
    """Tag contents are not escaped; the raw Observation is recorded beside it."""
    return OBSERVATION_TEMPLATE.format(
        status=observation.status,
        exit_code=observation.exit_code,
        stdout=_line_terminated(observation.stdout),
        stderr=_line_terminated(observation.stderr),
    )


def render_invalid(action: InvalidAction) -> str:
    return INVALID_TEMPLATE.format(reason=action.reason)


def is_verification(action: Bash, feedback_unchanged: bool) -> bool:
    return feedback_unchanged and _VERIFICATION.fullmatch(action.command) is not None


def _definition() -> dict[str, str]:
    return {
        "name": PROTOCOL_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": "task.prompt verbatim",
        "observation_format": OBSERVATION_TEMPLATE
        + " (stdout/stderr get a final newline if missing; not escaped)",
        "invalid_action_format": INVALID_TEMPLATE,
        "history": (
            "append-only token context: encoding.prompt(system, user), then per "
            "step the raw sampled tokens followed by encoding.continuation("
            "tokens, feedback); earlier tokens are never decoded and re-encoded"
        ),
        "verification": VERIFICATION_RULE,
    }


PROTOCOL_SHA256 = hashlib.sha256(
    json.dumps(_definition(), sort_keys=True).encode()
).hexdigest()


def describe() -> dict[str, str]:
    """The protocol as recorded in every run, with its fingerprint."""
    return _definition() | {"sha256": PROTOCOL_SHA256}
