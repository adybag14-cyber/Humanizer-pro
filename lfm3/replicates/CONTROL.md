# Same-seed training control

A third seed-57301 run completed on 25 November 2026 using the current trainer, the pinned LiquidAI base, the same Python environment and the original 48 training pairs. It is a reproducibility control outside the eight-candidate ranking. The historical generator produces exactly the same 48 source-target pairs as the current JSON file, in the same initial order.

The current reproduction and this control differ in all 188 trained tensors. Only 38 of 6,189,056 scalar values are exactly equal. The maximum absolute difference is 0.000447159, the mean absolute difference is 0.0000202500 and the RMS difference is 0.0000279429. Comparing tensor values establishes a numerical difference beyond file metadata. The delivery output is identical, while the meter and ticket outputs differ.

The historical adapter and control also differ in all 188 tensors. Their maximum absolute difference is 0.000364714. All three historical versus control outputs differ. The machine-readable comparison preserves full-precision measurements and all three weight hashes.

The original recipe seeds Torch and the shuffle. The trainer does not configure CUDA deterministic-algorithm settings. The measurements establish that the current recipe is insufficient for bitwise repeatability in these runs. The specific operation responsible for the variation has not been isolated. The results cannot identify all observed errors as random or exclude systematic errors.

The control used 48 optimizer steps. Its first-eight and last-eight mean training losses were 0.995930 and 0.00285755. Its 24,781,056-byte adapter has SHA-256 `6bd56d239af5a87dba2bc2a238dcea75c2510507e8acfc0cba8ba5abe8e9542e`.

- [Exact tensor comparisons](control-57301-comparison.json)
- [Control training record](control-57301.json)
- [Control outputs](control-57301-outputs.json)
- [First reproduction outputs](reproduction-57301-outputs.json)
- [Comparison script](compare_control.py)
