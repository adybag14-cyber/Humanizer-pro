# LFM3-1.7B editing experiment

The Fidelity release starts from LiquidAI’s instruction-tuned BF16 checkpoint and learns to shorten procedural notices. It ships in two forms. The original LoRA adapter contains 24,781,056 bytes. The standalone model merges the adapter into the BF16 base and can load directly in Transformers. [Download the release](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Fidelity/tree/79ccfcea39ff768642c7e323e926bad979513a82)

## Fine-tuning method

1. Start with the pinned [LiquidAI checkpoint](https://huggingface.co/LiquidAI/LFM3-1.7B/tree/0dc4268b561499c648c7ba7c9466bf3d8d01517a). The base weights have SHA-256 `135922caef64e81e52af88ede8b7c3ddaa9d6da220a9737fb3065275dbfb4330`.
2. Use all [48 synthetic training pairs](training-pairs.json). They vary the numbers and roles in three templates covering storage limits, temperature alerts and missing survey responses. The three evaluation passages are separate. This design tests whether a model can learn a narrow editing task from a small repetitive corpus.
3. Format the source using the base model’s chat template with thinking disabled. Compute next-token cross-entropy on the target edit and its end token. Mask all prompt tokens. Freeze base weights and train rank-8 LoRA on `q_proj`, `v_proj`, `w1`, `w2` and `w3`, with alpha 16 and zero dropout.
4. Shuffle with seed 57301. Use batch size one, one epoch, AdamW at 0.00005 and gradient-norm clipping at 1.0. Save the final adapter after 48 updates. Teacher-forced loss averaged 0.996 on the first eight examples and 0.00272 on the last eight. Both numbers describe training examples. [Run record](evidence/training.json)
5. Generate one edit per input using greedy decoding, a 384-token output limit and the same instruction for each model. Review each output against the seven source facts listed for that notice. [Inputs and factual checks](evidence/cases.json)

## Reproduce

The original environment used Python 3.12 on Ubuntu under WSL2, an RTX 4090, PyTorch 2.13.0+cu130 and Transformers 5.22.2. With a compatible CUDA build of PyTorch installed, run these commands from the repository root. The training command writes its adapter and evidence to a new directory.

```sh
python -m pip install -r lfm3/requirements.txt
python lfm3/train.py runs/fidelity
python lfm3/edit.py lfm3/evidence/observatory-source.txt --revision 79ccfcea39ff768642c7e323e926bad979513a82
```

## Editing results

| Test | Original base | Original base with LoRA | Merged BF16 checkpoint |
| --- | --- | --- | --- |
| Thursday deliveries | Copied source | Kept the inclusive 12 kg limit | Kept the inclusive limit |
| Electricity meters | Copied source | Kept the missing reading distinct from zero, but made the planned visit sound certain | Kept missing distinct from zero, also made the visit sound certain |
| Ticket refunds | Copied source | Omitted what starts the ten-working-day deadline and implied no cash deadline exists | Restored the deadline trigger but said the hall had not set a cash deadline |

The source says the hall has **not announced** a cash deadline. It may already have set one. The merged model changed the wording of all three greedy generations relative to the original LoRA inference path. [Base outputs](evidence/lfm3-base.json) · [Original adapter outputs](evidence/lfm3-fidelity.json) · [Merged model outputs](evidence/lfm3-merged.json)

## Fixed detector evaluation

On 25 November 2026 the merged model edited the fixed observatory passage from 220 to 191 whitespace-delimited words. The passage and checkpoint were fixed before a single submission to each service. GPTZero 5.0b assigned 100% document probability to AI, 0% to mixed and 0% to human. Pangram 4.1 counted 193 words and assigned the entire text to AI. The two services measure different quantities. One passage cannot establish a detector pass rate. [Source](evidence/observatory-source.txt) · [Exact edit](evidence/observatory-edit.txt) · [Readings and input hash](evidence/observatory-scans.json)

An earlier experiment used a different adapter, `lfm-v2`, with two inference passes. Its frozen output reached the joint threshold on **0/6** development passages (90% on each service) and **0/3** holdout passages (80% on each). The earlier results include each passage, its output, the two timestamped readings and the text hashes. [Earlier experiment records](evidence/prior-v2-evaluation.json)
