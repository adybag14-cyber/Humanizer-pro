# Frozen factual editing comparison

Eight original English notices cover booking conditions, refunds, responsibility, inclusive limits and uncertainty. Six require a real edit. Two are already clear, so leaving those passages alone is acceptable. The fixtures and grading rules were written without access to candidate training pairs or model outputs.

Each source claim receives one semantic judgment from the four values in `protocol.json`. Reviewers also list unsupported assertions, score clarity from 0 to 2 and judge whether the specified editing goal was met. The runner records normalized input copying and token overlap as descriptive checks. A copied passage can pass a clear control and cannot meet a required editing goal.

The ranking first counts fully acceptable cases, then minimizes fidelity failures, then maximizes required editing goals met and total clarity. A fully acceptable case preserves every source claim, adds no unsupported assertion, completes generation, meets its editing goal and scores at least 1 for clarity. Every candidate needs a complete review before ranking. Ties remain ties.

Eight methods on these eight passages cannot establish whether an error is random or systematic. Repeated seeds for a fixed recipe estimate training variation. Repeated greedy inference checks output repeatability. Shared failures across methods are reported by claim and case.

## Generate

Create a JSON manifest containing `base`, optional `base_revision`, and `candidates`. Each candidate has a unique `id` and either an `adapter` path or a `model` path. An empty candidate with only `id` evaluates the base. The runner loads the base once, loads each adapter onto that same base in turn, unloads it, then evaluates any merged checkpoints separately.

```sh
python lfm3/comparison/run.py manifest.json /path/to/new-output-directory
```

The output directory must be new. It contains one output and one review template per candidate, generation hashes, the fixture hash, runtime versions, and the manifest used. Fill review templates by comparing each output with its claims and editing goal.

```sh
python lfm3/comparison/rank.py /path/to/output-directory
```

Preserve the initial freeze file when adding later results.
