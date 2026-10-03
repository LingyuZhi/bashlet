"""Independent acceptance check of original TRIO deltas against merged weights."""
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

torch.set_num_threads(8)
manifest_path = Path(sys.argv[1])
record = json.loads(manifest_path.read_text())
base = Path(record["base_path"])
merged = Path(record["adapter"]["merged"]["path"])
source = Path("runs/issue21-local-inference/sft-step12/adapter/adapter_model.safetensors")
base_index = json.loads((base / "model.safetensors.index.json").read_text())["weight_map"]
merged_index = json.loads((merged / "model.safetensors.index.json").read_text())["weight_map"]
text = json.loads((base / "config.json").read_text())["text_config"]
key_size = text["linear_num_key_heads"] * text["linear_key_head_dim"]
checks = []
with ExitStack() as stack:
    original = stack.enter_context(safe_open(source, framework="pt"))
    readers = {}
    for folder, index in [(base, base_index), (merged, merged_index)]:
        for filename in set(index.values()):
            readers[str(folder / filename)] = stack.enter_context(safe_open(folder / filename, framework="pt"))

    def weight(folder, index, name):
        return readers[str(folder / index[name])].get_slice(name)

    for a_key in sorted(key for key in original.keys() if key.endswith(".lora_A.weight")):
        module = a_key.removesuffix(".lora_A.weight").removeprefix("base_model.model.")
        b_key = a_key.removesuffix(".lora_A.weight") + ".lora_B.weight"
        a = original.get_tensor(a_key).float()
        b = original.get_slice(b_key)
        rows = sorted({0, b.get_shape()[0] // 2, b.get_shape()[0] - 1})
        selected_b = torch.cat([b[row:row+1, :] for row in rows]).float()
        target = module + ".weight"
        offset = 0
        if module.endswith((".linear_attn.in_proj_q", ".linear_attn.in_proj_k", ".linear_attn.in_proj_v")):
            label = module[-1]
            target = module.rsplit("_", 1)[0] + "_qkv.weight"
            offset = {"q": 0, "k": key_size, "v": 2 * key_size}[label]
        base_target = "model.language_model.embed_tokens.weight" if module == "lm_head" else target
        source_weight = weight(base, base_index, base_target)
        destination_weight = weight(merged, merged_index, target)
        before = torch.cat([source_weight[row+offset:row+offset+1, :] for row in rows])
        actual = torch.cat([destination_weight[row+offset:row+offset+1, :] for row in rows])
        expected = (before.float() + 2 * (selected_b @ a)).to(before.dtype)
        error = (actual.float() - expected.float()).abs()
        upper = torch.nextafter(expected, torch.full_like(expected, float("inf")))
        lower = torch.nextafter(expected, torch.full_like(expected, -float("inf")))
        ulp = torch.maximum((upper.float()-expected.float()).abs(), (expected.float()-lower.float()).abs())
        within_ulp = bool(torch.all(error <= ulp))
        checks.append({"module": module, "rows": rows, "sampled_entries": actual.numel(), "exact_entries": int((actual == expected).sum()), "changed_from_base": int((actual != before).sum()), "max_abs_error": float(error.max()), "within_one_bf16_ulp": within_ulp})
        if not within_ulp:
            raise ValueError(f"Independent merge mismatch: {module}")
    embedding = "model.language_model.embed_tokens.weight"
    embedding_unchanged = torch.equal(weight(base, base_index, embedding)[:], weight(merged, merged_index, embedding)[:])
    assert embedding_unchanged

result = {"passed": True, "snapshot_manifest_sha256": hashlib.file_digest(manifest_path.open("rb"), "sha256").hexdigest(), "reference": "original split TRIO tensors, alpha/r=32/16, independent Torch FP32 matmul then BF16 cast", "scope": "three sampled rows per original target; complete input embedding equality", "torch_version": torch.__version__, "original_target_modules": len(checks), "input_embedding_unchanged": embedding_unchanged, "checks": checks}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2))
print({key: result[key] for key in ("passed", "scope", "original_target_modules", "input_embedding_unchanged")})
