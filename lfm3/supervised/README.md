# Independent supervised editing candidates

Both candidates begin from LiquidAI/LFM3-1.7B revision `0dc4268b561499c648c7ba7c9466bf3d8d01517a`. Their synthetic training pairs and settings were frozen before training. The exact file hashes and freeze time are in [freeze.json](freeze.json).

| Candidate | Data | LoRA targets | Rank / alpha | Learning rate | Seed |
| --- | --- | --- | --- | --- | --- |
| edit-keep | 16 clear passages paired with themselves and 16 clarity rewrites | q_proj, v_proj, w1, w2, w3 | 8 / 16 | 0.00005 | 57421 |
| minimal | 32 individual corrections covering agreement, redundancy and punctuation | q_proj, v_proj | 8 / 16 | 0.00003 | 57422 |

Each run uses BF16 base weights, one epoch, one example per step, AdamW with its PyTorch defaults except the specified learning rate, zero LoRA dropout and gradient clipping at 1.0. The prompt comes from the common editing runtime. Loss applies to target tokens and the end token. The inputs receive no loss. The run records include hashes of the data, settings, script and saved adapter.

These small pilot datasets test different editing objectives. The edit-keep candidate tests whether clear passages remain intact while verbose passages receive edits. The minimal candidate tests a narrow correction objective with a smaller set of trainable layers. Their shared origin and training size limit claims about broad writing quality.

```sh
python lfm3/supervised/train.py edit-keep /models/first-release-edit-keep --base-dir /models/LFM3-1.7B
python lfm3/supervised/train.py minimal /models/first-release-minimal --base-dir /models/LFM3-1.7B
```

The output directories must be new. Each contains the PEFT adapter, portable base-model configuration, training record and process ownership record. Use the pinned dependency versions in `lfm3/requirements.txt`.
