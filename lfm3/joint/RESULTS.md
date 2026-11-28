# LFM3-1.7B Humanizer Pro results

The published three-adapter model returned one rewrite per passage. On **34 of 57** passages (59.6%), GPTZero reported 100% human document probability and Pangram reported 100% human text coverage on the same rewrite. GPTZero gave its maximum reading on 36/57 rewrites and Pangram on 45/57. Of the 56 original passages that fell short on at least one detector, 33 passed both after rewriting. [Per-passage scores and text hashes](evidence/detector-readings.json) · [Benchmark scorecard](evidence/benchmark-summary.json)

| Configuration | Historical | First comparison | Reserved confirmation | Total |
| --- | ---: | ---: | ---: | ---: |
| Starting model | 5/25 | 8/16 | 5/16 | 18/57 |
| Feedback-trained adapter alone | 11/25 | 11/16 | 7/16 | 29/57 |
| Two-adapter release | 9/25 | 13/16 | 10/16 | 32/57 |
| **Published three-adapter release** | **9/25** | **14/16** | **11/16** | **34/57** |

The first 41 passages informed model selection. The output hashes were fixed before reviewing the results for the 16 reserved passages. A four-adapter configuration tied the published model on every panel, so the release uses three adapters. [Frozen confirmation hashes](evidence/confirmation.json) · [Source overlap audit](evidence/source-freeze.json)

## Repeated runs

Three historical and three first-comparison passages were chosen by source hash. The published model passed on **3/6** at its default seed (27183), **1/6** at seed 28151 and **3/6** at the follow-up seed 29411. Across those 18 runs, seven passed both detectors and four distinct passages passed at least once. The earlier two-adapter model passed eight of its 18 runs on the same inputs. [Three-adapter seed readings](evidence/seeds-voice-mix.json) · [Two-adapter seed readings](evidence/seeds-mix-v2.json) · [Follow-up seed replay](evidence/follow-up-seed-replay.json)

## Factual accuracy

Each of 12 passages had five prespecified facts to check. The published model retained **26/60**, compared with **31/60** for the starting model. On two additional inputs tried after model selection, the bus-tally rewrite scored 100% on both detectors while changing a proposed trial bus service into people being put on trial. The recipe-card rewrite scored 68% human on GPTZero and 100% human coverage on Pangram. These two inputs are reported separately from the 57-passage benchmark. [Fact checks](evidence/benchmark-summary.json) · [Additional inputs and exact rewrites](evidence/post-release-smoke.json)

The model weights, runnable decoder and example are available at the verified Hugging Face revision. All 39 files listed in its manifest matched the anonymous download, and the published decoder reproduced the 216-word example byte-for-byte. [Model and inference instructions](https://huggingface.co/Tdamre/LFM3-1.7B-Humanizer-Pro/tree/2cde52c70bd4e10d48cd66048828facb92560db5) · [Publication receipts](evidence/releases.json)
