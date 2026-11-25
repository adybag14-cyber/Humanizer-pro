# Preference training for factual editing

This pilot starts from the pinned LiquidAI LFM3-1.7B checkpoint. Each of its 32 original synthetic examples has a source passage, a preferred concise edit and a rejected edit containing a specific factual error. The examples cover reservation boundaries, missing readings, uncertain dates, delivery conditions, responsibility for approvals, strict inequalities, future availability and capacity that includes guests. The complete triples are in [pairs.json](pairs.json).

Direct preference optimization compares the preferred and rejected completion log probabilities against a frozen reference. The reference is the original base checkpoint, evaluated with the LoRA adapter disabled. Reference values are cached before optimization. The loss is the negative log sigmoid of beta times the reference-adjusted preferred-minus-rejected log-probability difference. Probabilities are summed over completion tokens, including the end token. Prompt and padding tokens contribute no loss.

The fixed settings use LoRA rank 8, alpha 16, zero dropout and the q_proj, v_proj, w1, w2 and w3 modules. AdamW uses a learning rate of 0.00002 and gradient clipping at 1.0. Beta is 0.1. Two epochs produce 64 optimizer updates with one preferred/rejected pair per update. The shuffle seed is 41927. The experiment retains its final checkpoint. Settings, source and data hashes are written before model loading and copied into the completed training record.

```sh
python lfm3/preference/train.py /models/factual-preference-run --base-dir /models/LFM3-1.7B
```

Use the dependencies in the parent LFM3 directory. The output contains a portable PEFT adapter, the completed training record, the frozen plan and a local process ownership record. The training record includes the adapter hash and peak allocated CUDA memory. The data and settings were fixed before the pilot launched.
