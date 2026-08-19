# B200 standings

`run_bench.sh` with `REPS=5`, 8 variants × 15 shapes. Times are the fastest of
5 repeats; `copy` is the no-reduction, no-quantize ceiling at the same byte
traffic. Every shape is measured against cold data — iterations rotate through a
5 GiB pool, so a shape whose own footprint fits in L2 is not measured out of it.

## Who wins where

Gap is to `copy` at that shape.

| C | N=1024 | N=8192 | N=65536 |
|---|---|---|---|
| 1024 | `fixed_32pt` 1.68x * | **`fixed_32pt` 1.06x** | `fixed_32pt` 1.11x |
| 2048 | `fixed_c` 1.31x * | `pers_32pt` 1.15x | `fixed_32pt` 1.13x |
| 4096 | three-way tie 1.32x | `tma_simple_swz` 1.17x | `tma_simple_swz` 1.18x |
| 8192 | `fixed_32pt` 1.20x | `epi` 1.12x | `epi` 1.14x |
| 16384 | `tma_simple_swz` 1.26x | `epi` 1.19x | `epi` 1.20x |

\* the N=1024 rows at C ≤ 2048 carry 9-14% spread and are not a ranking; see
"Where the numbers stop meaning anything".

Three kernels split the space and the boundary is C:

- **`fixed_32pt`** — C ≤ 2048 at every useful N, and C=8192 when the grid is
  small. 4 wins.
- **`tma_simple_swz`** — C=4096 outright, and C=16384 at N=1024. 3 wins.
- **`epi`** — C ≥ 8192 once the grid fills the machine, and the only kernel that
  runs above C=8192 apart from `loop` and the TMA pair. 4 wins.
- **`pers_32pt`** — one win, C=2048/N=8192, by 1.1%.
- **`loop`**, **`fixed_c`**, **`tma_simple`** — no wins at any shape.

Best absolute result: **`epi` at 6267 GB/s**, C=8192/N=65536, 1.14x off the
ceiling. Best relative: **`fixed_32pt` at 1.06x**, C=1024/N=8192.

## Full results

GB/s, higher is better. Bold is the best kernel at that shape.

### N=65536 — the throughput regime

| kernel | C=1024 | C=2048 | C=4096 | C=8192 | C=16384 |
|---|---|---|---|---|---|
| `copy` (ceiling) | 6721 | 6972 | 7056 | 7119 | 7113 |
| `loop` | 3216 | 4437 | 5025 | 5528 | 5286 |
| `epi` | 3724 | 4919 | 5579 | **6267** | **5934** |
| `fixed_c` | 5633 | 5753 | 5456 | 4211 | — |
| `fixed_32pt` | **6073** | **6177** | 5838 | 5293 | — |
| `pers_32pt` | 5921 | 6039 | 5828 | 5181 | — |
| `tma_simple` | 4428 | 5950 | 5631 | 5542 | 5001 |
| `tma_simple_swz` | 4251 | 5967 | **5955** | 5566 | 5753 |

### N=8192

| kernel | C=1024 | C=2048 | C=4096 | C=8192 | C=16384 |
|---|---|---|---|---|---|
| `copy` (ceiling) | 4101 | 5370 | 6166 | 6543 | 6773 |
| `loop` | 2304 | 3670 | 4313 | 5121 | 5170 |
| `epi` | 2580 | 3880 | 4817 | **5846** | **5682** |
| `fixed_c` | 3419 | 4548 | 4846 | 4263 | — |
| `fixed_32pt` | **3860** | 4552 | 5223 | 4963 | — |
| `pers_32pt` | 3721 | **4655** | 5157 | 4741 | — |
| `tma_simple` | 3164 | 4385 | 5107 | 5086 | 4889 |
| `tma_simple_swz` | 2927 | 4507 | **5292** | 5268 | 5349 |

### N=1024 — an underfilled machine

| kernel | C=1024 | C=2048 | C=4096 | C=8192 | C=16384 |
|---|---|---|---|---|---|
| `copy` (ceiling) | 1149 | 2030 | 3374 | 4096 | 5120 |
| `loop` | 636 | 1281 | 2045 | 2282 | 2596 |
| `epi` | 653 | 1384 | 2049 | 2657 | 3152 |
| `fixed_c` | 674 | 1554 | 2558 | 2923 | — |
| `fixed_32pt` | 685 | 1508 | 2560 | **3409** | — |
| `pers_32pt` | 654 | 1307 | 2555 | 2962 | — |
| `tma_simple` | 535 | 1062 | 2073 | 2926 | 3724 |
| `tma_simple_swz` | 525 | 1054 | 2060 | 2925 | **4049** |

## Notes

- **Swizzle earns its place above C=2048 only.** `tma_simple_swz` vs plain
  `tma_simple`: −13% at C=16384/N=65536, −4% at C=4096, but +4% at C=1024. It is
  what makes the TMA lineage competitive at large C and what sinks it at small C.
  (`HANDOFF.md` says the swizzle is noise — measured at C=2048, the one shape
  where that holds.)
- **Register-resident designs stop scaling at C=8192.** `fixed_c` 1.69x off the
  ceiling there, `fixed_32pt` 1.34x, both leading below C=4096. The live state
  stops fitting and two-pass `epi` takes over.
- **`pers_32pt` does not pay for itself.** One win by 1.1%; the weight reuse is
  cancelled by occupancy, 31-33% against rung 03's 38-39%.
- **6-20% headroom left**, tightest at small C.

## Caveats

The N=1024 column is not a bandwidth measurement — `copy` itself only reaches
1149 GB/s at C=1024 against 6721 at N=65536. With 1024 blocks over 148 SMs the
machine is underfilled and launch overhead is comparable to the kernel; spreads
run 9-14% where every other row is under 2%. Read it as tolerance of an
underfilled machine, not a ranking.

The C ≤ 8192 rungs `SKIP` at C=16384: their launchers `std::exit` past that
bound.
