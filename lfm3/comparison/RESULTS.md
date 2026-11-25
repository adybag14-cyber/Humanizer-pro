# Eight LFM3-1.7B first releases

**Minimal ranked first on the eight-notice editing test.** Its 2.35 MB attention adapter preserved all 64 checked facts and made three successful edits. It copied the other five inputs, including three that asked for a change. Seed 57303 also delivered five fully acceptable passages. It ranked second because it lost facts and one generation reached the output limit.

## Results by release

| Rank | Pinned Hugging Face release | Fully acceptable / 8 | Claims preserved / 64 | Required edits achieved / 6 | Inputs copied / 8 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | [Minimal](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Minimal/tree/da35d3d0d3f26df3e5cb507d42ea8f04b469f3a7) | 5 | 64 | 3 | 5 |
| 2 | [Seed-57303](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Seed-57303/tree/ad19ca4c1d15f6cc9d23e9e37d0148cc8a09325a) | 5 | 54 | 3 | 0 |
| 3 | [Seed-57302](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Seed-57302/tree/b18f5ead5bb30fdd17ae1f310598f6c92612a7be) | 4 | 58 | 5 | 0 |
| 4 | [Fidelity](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Fidelity/tree/79ccfcea39ff768642c7e323e926bad979513a82) | 3 | 56 | 3 | 1 |
| 5 | [Weighted](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Weighted/tree/5009f1fa490e1a10278927a43a5bd297f9a73ffe) | 3 | 55 | 3 | 2 |
| 6 | [Edit-Keep](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Edit-Keep/tree/8c1354bd720f2104e6592b3b1e4eec53027319fc) | 2 | 64 | 0 | 8 |
| 7 | [Unlikelihood](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-Unlikelihood/tree/af85d77b018e41def74e49a8f5771a11f0890e7c) | 2 | 60 | 3 | 0 |
| 8 | [DPO](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro-DPO/tree/9cdc628a9c47c98ac941e423c2416c7e6cf1fed8) | 1 | 56 | 2 | 0 |

The untuned LiquidAI control scored 2/8 acceptable, preserved 63/64 checked claims, met 0/6 required editing goals and copied seven inputs. Copies are counted after whitespace normalization. An unchanged passage can pass either of the two already-clear cases. It cannot pass an edit-required case.

Six training methods contributed eight candidates. Fidelity and Seeds 57302 and 57303 use the same recipe and training pairs. The other five candidates use direct preference optimization, edit-or-keep supervision, minimal correction, weighted loss on factual spans, or unlikelihood training on incorrect factual tokens. Several methods also differ in training data and LoRA targets. The results compare the trained models as released. [Per-model training details and revisions](releases.json)

## Which errors repeated?

- On the ceramics notice, five completed rewrites turned *has not completed firing* into *has not been fired*. Firing could already be in progress. These were Fidelity, DPO, Weighted, Unlikelihood and Seed 57302. Seed 57303 separately reached the 384-token limit on this case and returned no completed edit.
- On the language-class notice, five candidates removed which organization’s membership was optional. All three seeds of the original Fidelity recipe made that error. The source specifies the community centre, while the notice also mentions a conversation club.

These recurrences provide evidence of shared failure patterns on the test set. The three Fidelity seeds scored 3/8, 4/8 and 5/8 fully acceptable. Both the method and training variation can matter. Eight short English notices cannot establish their frequencies in other genres. [Claim-by-claim results](results/summary.json)

## Reproducibility and review

The input passages, factual checks, decoding settings and ranking rule were frozen before generation. All eight candidates used the same pinned base and were evaluated as PEFT adapters to keep weight representation constant. The original release also ships a merged BF16 checkpoint whose output can differ after rounding. Four model reviewers each assessed two cases. They saw the source, editing goal and candidate texts with anonymous identifiers. Identical text in a case received a single judgment shared across all models that produced it. This was a model-assisted review with no independent human grader. The saved reviews give the reason for every failed or uncertain claim. [Frozen protocol](protocol.json) · [Sources and checks](cases.json) · [Outputs and reviews](results/)

Two fresh runs of greedy inference from the saved weights produced 63 identical completed candidate passages and the same token-limit failure on the 64th. All eight outputs from the untuned control repeated as well. Repeating **training** had a different result. Two runs with identical seed, code and data differed in all 188 trained tensors and produced different text on two of three historical cases. The precise operation responsible remains unisolated. [Same-seed comparison](../replicates/CONTROL.md)

An additional cold run loaded the publicly downloaded base and all eight pinned adapters. Every completed edit matched the original output bytes and the same ceramics failure recurred. The documented Minimal CLI also reproduced the exact observatory text used for the live check. [Public artifact replay](results/public-reproduction.json)

## Fixed live detector comparison

On 25 November 2026, **all eight releases received 100% AI from both services** on the same observatory scenario. GPTZero 5.0b reported a document probability of 100% AI, 0% mixed and 0% human for every submission. Pangram 4.1 marked the whole text as AI-generated each time. The original Fidelity submission used the merged BF16 checkpoint. The other seven used their published PEFT adapters.

The original pair comprised Fidelity’s 191-word edit and Minimal’s verbatim copy of the 220-word source. A follow-up covered the remaining six releases. All six outputs were generated with the original prompt and greedy decoder and frozen before the first follow-up scan. Each output was submitted once per dashboard, including the identical copies from Edit-Keep and Weighted. There were six distinct texts across all eight submissions to each service. The editing-quality ranking was fixed before the follow-up. These results cover one passage per model. [Original Fidelity readings](../evidence/observatory-scans.json) · [Minimal readings](results/observatory-scans.json) · [Follow-up output hashes](results/observatory-six/outputs.json) · [Twelve follow-up readings](results/observatory-six/readings.json)

| Follow-up model and submitted text | Words | Exact source copy | GPTZero AI probability | Pangram AI coverage |
| --- | ---: | --- | ---: | ---: |
| [DPO](results/observatory-six/preference.txt) | 230 | No | 100% | 100% |
| [Edit-Keep](results/observatory-six/edit-keep.txt) | 220 | Yes | 100% | 100% |
| [Weighted](results/observatory-six/weighted.txt) | 220 | Yes | 100% | 100% |
| [Unlikelihood](results/observatory-six/unlikelihood.txt) | 228 | No | 100% | 100% |
| [Fidelity, seed 57302](results/observatory-six/seed-57302.txt) | 191 | No | 100% | 100% |
| [Fidelity, seed 57303](results/observatory-six/seed-57303.txt) | 188 | No | 100% | 100% |

## Reproduce the comparison

With the CUDA Python environment described in the [training method](../README.md), run these commands from the repository root. Fetch checks all eight adapter weight hashes. The public manifest pins the base and each adapter. The rank command recomputes the editing order from the committed semantic reviews.

```sh
python lfm3/comparison/fetch.py comparison-models.json
python lfm3/comparison/run.py comparison-models.json runs/comparison
python lfm3/comparison/rank.py lfm3/comparison/results
```
