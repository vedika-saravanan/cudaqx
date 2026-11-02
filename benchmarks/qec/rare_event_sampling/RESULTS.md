# LACE results: layer-stratified allocation with certified error bars against direct Monte Carlo and the IBM baseline

All numbers come from the JSON reports in `results/` with the stated seeds;
nothing is tuned after the fact. Hardware and software are listed in
`README.md`. "Direct" always means an independent Monte Carlo of the same
circuit decoded by the same decoder instance, the ground truth wherever it
is affordable. Codes: BB72 (`[[72,12,6]]`, 6 rounds, reference rate
p = 2e-3) with Relay-BP (GPU, batch 1024) and with `ldpc` BP-OSD (CPU,
min-sum, 100 iterations, OSD-CS order 10); surface code d = 5, 5 rounds,
p = 3e-3, with PyMatching.

## 1. The three estimators

**Direct Monte Carlo.** Sample the circuit at the target rate p with Stim's
circuit sampler, decode every shot, count failures. Unbiased; the cost to a
relative error e is 1 / (e^2 LER), so 2.3e8 decodes for 10% at LER 4e-7,
and every rate needs its own run.

**IBM "Fail fast" (count strata).** Write LER(p) = sum_k P_k(p) f(k) with
P_k the exact Poisson-binomial mass of k faults and f(k) = P(fail | k)
measured by exact conditional sampling of each count stratum once. The
measured f(k) are reweighted exactly to every target rate (the conditional
law of the subset given k depends on the odds ratios only, and the exact
likelihood ratio is applied where they change). IBM's paper additionally
fits an ansatz to f(k) at low k and extrapolates; that fit is implemented
and measured in section 2 and is biased on BB72, so the unbiased reweighted
variant is the baseline used throughout.

**LACE, Layer-stratified Allocation with Certified Error bars (this work).**
Inside each count stratum the configurations are split exactly by the
composition of the k faults over the detection layers (the mass of each
composition is a product of per-layer Poisson-binomial terms, so every
group weight is exact), grouped by a position statistic: for compact sets
(span at most 2 layers) the exact first and last layer, for spread sets the
start and count of the densest 2-layer window; `position_core` refines by
the largest single-layer count. Groups below 0.2% of the stratum are
pooled into one sampled "rest" group, so nothing is dropped. Allocation:
a mass-proportional pilot of 10% of the budget, then three Neyman rounds
whose rates are shrunk hierarchically toward the pooled rate of the
group's position-free family; rounds are combined with weights fixed in
advance, which keeps the estimator unbiased under adaptive allocation, and
the variance is the per-round unbiased plug-in. Certification: a group that
never fails contributes zero to the plug-in variance but a rate up to
3 / N_f cannot be excluded for its family after N_f failure-free labels,
so the *certified* variance charges each never-failing family that
rule-of-three rate. Inside the rounds the never-failing families receive
enough labels that the sum of those charges is at most an allowance
epsilon times the plain-sampling variance at the planned budget, split
across families in proportion to W_f^(2/3) (the minimum-label split); the
allowance is either fixed (the default, 0.15) or chosen each round as the value that
minimises the projected certified variance, Neyman plug-in on the labels
the failing groups keep plus the bound the empty ones are allowed
(`--certify-fraction auto`; section 6 measures both). The gain is reported two ways: *plug-in*
(variance of plain conditional sampling at the same decode count over the
estimator's plug-in variance; what the estimator realises) and *certified*
(the same with the certified variance; what one run can promise with no
assumption about the never-failing groups).

The same stored spectrum answers every target rate; the layer gains per
count enter the LER-level cost through the exact count masses.

## 2. The IBM baseline on its own terms

Surface code d = 5, PyMatching: 21 strata (k = 0..20), 20,000 exact
conditional samples each (420,000 decodes, 13.6 s). All 50 reduced failing
sets have weight 3, so IBM's enclosure model holds here.

| p | Exact stratified | Reweighted (unbiased) | f3 ansatz | f5 ansatz | Direct (400k shots) |
|---|---|---|---|---|---|
| 0.003 | 3.20e-3 [2.54e-3, 4.18e-3] | - | 4.66e-3 | 3.21e-3 | 3.25e-3 [3.08e-3, 3.43e-3] |
| 0.001 | - | 1.42e-4 +- 0.14e-4 | 2.80e-4 | 1.37e-4 | 1.13e-4 [0.82e-4, 1.51e-4] |
| 0.0003 | - | 4.4e-6 +- 0.6e-6 | 9.5e-6 | 4.0e-6 | 1.0e-5 (4 failures) [2.7e-6, 2.6e-5] |

BB72, Relay-BP: 25 strata (k = 0..24), 20,000 samples each (500,000
decodes, 48 s), then a second pass with IBM's allocation (k = 3..8,
300,000 samples each, 1.8M decodes, 128 s). Minimal failing sets found by
descent from 40 observed failures have weights 3 (19), 4 (11), 5 (9) and
6 (1): many failures are irreducible at weight 4-6, so the enclosure model
behind the ansatz is violated for Relay-BP on BB72.

| p | Reweighted (unbiased) | f3 ansatz | Direct (400k shots) |
|---|---|---|---|
| 0.002 | 9.56e-3 [8.14e-3, 1.13e-2] | 1.12e-2 | 9.34e-3 [9.04e-3, 9.64e-3] |
| 0.0015 | 3.29e-3 +- 0.10e-3 | 4.39e-3 | 3.32e-3 [3.15e-3, 3.51e-3] |
| 0.001 | 7.5e-4 +- 0.5e-4 | 1.24e-3 | 8.1e-4 [7.2e-4, 9.0e-4] |
| 0.0005 | 7.6e-5 +- 1.4e-5 | 1.56e-4 | 7.3e-5 [4.9e-5, 10.4e-5] |
| 0.0002 | 5.0e-6 +- 1.8e-6 | 1.08e-5 | 2.5e-6 (1 failure) [6e-8, 1.4e-5] |

Second pass (k = 3..8, 300k samples each): f(3) = 8.7e-5, f(4) = 4.3e-4,
f(5) = 1.45e-3, f(6) = 3.5e-3, f(7) = 7.3e-3, f(8) = 1.34e-2; k <= 2 are
non-failing by exhaustive enumeration of weight 1 and 2 for BP-OSD and
PyMatching and f(2) = 3.18e-7 exactly for Relay-BP (5 failing weight-2
pairs, `run_exact_head.py`).

| p | Reweighted from k = 3..8 | Relative SE | Direct shots for the same precision |
|---|---|---|---|
| 0.0005 | 7.00e-5 +- 0.29e-5 | 4% | 9e6 (1.8e6 used) |
| 0.0002 | 3.60e-6 +- 0.35e-6 | 10% | 2.8e7 |
| 0.0001 | 4.2e-7 +- 0.6e-7 | 14% | 1.3e8 |

The ansatz over-predicts by 1.2x at p = 2e-3 and 2.2x at p = 2e-4 and its
projection depends on allocation and model choice (f5 fits within 7 nats
of log-likelihood project 0.36x to 0.94x of the unbiased value at 2e-4)
far more than on its 2% bootstrap width. The exact reweighting is the
baseline; the ansatz is not used again.

## 3. LACE: certified gains per count

Each row is the estimator's gain over plain conditional sampling of the
same count stratum at the realised decode count (certification floors add
to the nominal budget; every comparison is at the realised count). The
"bound vs break-even" column gives the mass-weighted rule-of-three rate of
the never-failing groups against the rate at which their contribution
would cancel the gain.

| Decoder, k | configuration | decodes per run | plug-in gain | certified gain | bound vs break-even |
|---|---|---|---|---|---|
| Relay-BP, 3 | in-round, epsilon 0.3 | 0.9M | 3.7x | **1.44x** | 7.5e-6 vs 3.6e-5 |
| Relay-BP, 3 | in-round, epsilon 0.15 | 1.1M | 3.3x | **1.73x** | 5.9e-6 vs 4.1e-5 |
| Relay-BP, 3 | in-round, epsilon 0.15, 3x budget | 2.5M | 3.9x | **2.23x** | 5.3e-6 vs 3.7e-5 |
| Relay-BP, 4 | in-round, epsilon 0.3 | 0.65M | 1.66x | **1.14x** | 2.1e-5 vs 9.3e-5 |
| BP-OSD, 3 | in-round, epsilon 0.3 | 1.0M | 7.0x | **1.15x** | 8.8e-6 vs 2.0e-5 |
| BP-OSD, 3 | in-round, epsilon 0.3, 2x budget | 2.0M | 7.1x | **1.46x** | 5.1e-6 vs 1.5e-5 |
| BP-OSD, 3 | in-round, epsilon 0.15, 2x budget | 2.5M | 1.4x | 0.75x | 3.6e-6 vs 7.5e-6 |
| BP-OSD, 4 | position + core, in-round, epsilon 0.3 | 1.6M | 4.8x | **1.56x** | 1.1e-5 vs 4.4e-5 |
| PyMatching d=5, 3 | position | 0.6M | 1.44x | 1.03x | - |
| PyMatching d=5, 4 | position | 0.6M | 1.03x | 1.03x | - |

Reports: `results/layers_bb72_relay_position_inround_{eps015,eps03,eps015_1p8M,eps03_k4}.json`,
`results/layers_bb72_bposd_position_inround_{eps03,eps03_1p2M,eps015_1p2M}.json`,
`results/layers_bb72_bposd_position_core_inround_eps03_k4_1p2M.json`,
`results/layers_surface5_pymatching_position.json`.

The certified gain tracks the sqrt(N) scaling of the certification cost:
for Relay-BP at onset it is 1.4-1.7x at about 1M decodes and 2.2x at
2.5M, while the plug-in gain stays near 3.3-3.9x. BP-OSD has the largest
plug-in gain (7x) and the smallest certified one at 1M decodes because
88-97% of its onset mass never fails and f(3) is three times smaller, so
certifying costs the most; doubling the budget lifts it from 1.15x to
1.46x with the plug-in unchanged, as the 1 / N^2 fall of the bound
predicts. For matching on the surface code the layer structure is worth
1.03x certified at d = 5 and nothing certifiable at d = 7 (f(4) = 6.7e-6
puts the onset stratum behind a 1 / f wall; `results/layers_surface7_*`),
so count strata alone are the right tool there.

## 4. LER-level cost against both baselines

`run_ler_speedup.py` combines the k <= 8 spectrum (one 5e5-decode
measurement, reweighted exactly) with the per-count gains of section 3 and
reports the decoder calls each estimator needs for a 10% relative error at
every target rate. For Relay-BP the exact weight-2 head 3.18e-7 is added;
the Relay-BP row at p = 2e-3 is a lower bound (15% of the mass lies above
k = 8 there). Per-count gains used: Relay-BP k = 3 epsilon 0.15, k = 4
epsilon 0.3, k = 5 default run; BP-OSD k = 3 epsilon 0.3 at 2M, k = 4
position + core; PyMatching position runs.

| Decoder / code | p | LER | direct MC | IBM count strata | layer strata, plug-in | layer strata, certified | vs direct (plug-in) | vs IBM: plug-in / certified |
|---|---|---|---|---|---|---|---|---|
| Relay-BP, BB72 | 2e-3 | 3.2e-3 | 3.1e4 | 1.7e4 | 1.6e4 | 1.7e4 | 2x | 1.10x / 1.04x |
| Relay-BP, BB72 | 1e-3 | 6.7e-4 | 1.5e5 | 5.0e4 | 3.6e4 | 4.4e4 | 4x | 1.40x / 1.15x |
| Relay-BP, BB72 | 5e-4 | 7.0e-5 | 1.4e6 | 1.8e5 | 9.1e4 | 1.3e5 | 16x | 1.92x / 1.32x |
| Relay-BP, BB72 | 2e-4 | 3.6e-6 | 2.8e7 | 5.1e5 | 2.0e5 | 3.3e5 | 140x | 2.6x / 1.52x |
| Relay-BP, BB72 | 1e-4 | 4.3e-7 | 2.3e8 | 7.3e5 | 2.5e5 | 4.5e5 | 930x | 2.9x / 1.61x |
| BP-OSD, BB72 | 2e-3 | 9.0e-4 | 1.1e5 | 6.3e4 | 5.7e4 | 6.1e4 | 2x | 1.10x / 1.03x |
| BP-OSD, BB72 | 1e-3 | 1.9e-4 | 5.4e5 | 1.8e5 | 1.1e5 | 1.6e5 | 5x | 1.62x / 1.16x |
| BP-OSD, BB72 | 5e-4 | 1.9e-5 | 5.2e6 | 6.5e5 | 2.1e5 | 4.8e5 | 25x | 3.1x / 1.34x |
| BP-OSD, BB72 | 2e-4 | 1.0e-6 | 9.9e7 | 1.9e6 | 3.4e5 | 1.3e6 | 290x | 5.4x / 1.45x |
| BP-OSD, BB72 | 1e-4 | 1.2e-7 | 8.5e8 | 2.7e6 | 4.3e5 | 1.9e6 | 2000x | 6.4x / 1.46x |
| PyMatching, surface d=5 | 1e-3 | 1.4e-4 | 7.1e5 | 3.6e4 | 2.8e4 | 3.5e4 | 25x | 1.3x / 1.03x |
| PyMatching, surface d=5 | 3e-4 | 4.4e-6 | 2.3e7 | 5.2e4 | 3.7e4 | 5.0e4 | 620x | 1.4x / 1.03x |

(`results/ler_speedup_*.json`, `figures/ler_speedup.png`.) The advantage
over IBM grows toward low LER because the onset stratum's share of the
count-stratified variance rises from 2% at p = 2e-3 to 85% at p = 1e-4 on
BB72, and that is the stratum where the layer structure acts. The
layer-stratified cost rises only 1.3x between p = 2e-4 and 1e-4 while
direct Monte Carlo rises 8.5x.

## 5. Stress tests against direct Monte Carlo

**Direct controls at every rate.** `run_direct_mc.py` with the same
Relay-BP instance (Stim circuit sampler, an independent code path from the
DEM-based conditional samplers); the pipeline column is the reweighted
spectrum of section 4.

| p | direct MC LER (95% Clopper-Pearson) | failures / shots | wall | pipeline LER | decodes for 10%: direct / pipeline |
|---|---|---|---|---|---|
| 1e-3 | 7.20e-4 [5.91e-4, 8.69e-4] | 108 / 1.5e5 | 0.1 min | 6.67e-4 | 1.5e5 / 3.6e4 |
| 5e-4 | 7.14e-5 [5.81e-5, 8.69e-5] | 100 / 1.4e6 | 0.7 min | 7.01e-5 +- 0.29e-5 | 1.4e6 / 9.1e4 |
| 2e-4 | 3.71e-6 [3.03e-6, 4.50e-6] | 104 / 2.8e7 | 15 min | 3.63e-6 | 2.8e7 / 2.0e5 |
| 1e-4 | 4.58e-7 [3.77e-7, 5.52e-7] | 110 / 2.4e8 | 121 min | 4.26e-7 | 2.3e8 / 2.5e5 |

(`results/direct_mc_bb72_relay_*.json`.) The pipeline estimate is inside
the direct interval at every rate; at 1e-4 it is 7% below a direct point
estimate that has a 9.5% standard error.

**Ten independent repeats, Relay-BP k = 3** (epsilon 0.15, 600k budget,
1.07M decodes per arm; `..._eps015_10rep.json`). Pooled truth from the
plain samples of all Relay-BP position reports: f(3) = 9.19e-5 from
11,288 failures (0.9%). Stratified mean +0.0% from truth; realised
variance gain across repeats 3.35 [95% bootstrap 1.67, 6.10] against a
mean reported plug-in gain of 3.05 and certified 1.41 (per-repeat
certified 0.71-2.05, one below 1).

**Ten independent repeats, BP-OSD k = 3** (epsilon 0.3, 1.2M budget,
2.1M decodes per arm; `..._eps03_1p2M_10rep.json`). Stratified mean +3.0%
from a 1201-failure pooled truth, 1.0 sigma with the truth's own error.
Every repeat certified above 1 (1.19-1.68, median 1.41); plug-in 5.5-14.8
(median 7.3). The realised gain across ten repeats, 3.5 [0.83, 14.7], is
too wide to pin the plug-in figure better than a factor 2.

**Coverage, 100 repeats, Relay-BP k = 3** (epsilon 0.15, 200k budget,
5.1e5 decodes per arm, about 50 failures each; `..._eps015_coverage.json`,
`summarise_coverage.py`). The plug-in interval covered the pooled truth
69% and 95% of the time at the 68% and 95% levels; the certified interval
84% and 99%; the plain binomial interval 73% and 94%. The realised standard
deviation equals the mean reported plug-in standard error (ratio 1.00) and
is 0.68x the certified one. Realised gain 2.55 [1.71, 3.81] against 2.70
reported (plug-in) and 1.24 (certified; 14 of 100 repeats below 1 at this
budget). Stratified mean 1.8% below truth, 1.3 sigma.

What the three tests establish: the estimator is unbiased against an
independent control at every rate of the LER table; the plug-in standard
error is calibrated at 50-100 failures per arm and conservative by 1.3x
below that; the certified bound is always conservative; the certified
gains in the tables hold in every repeat but one of 120 (that one at the
smallest budget); the plug-in gains are what the estimator realises,
confirmed to 30% for Relay-BP and to a factor 2 for BP-OSD.

**Wall time.** Direct Monte Carlo ran at 30 us per shot end to end. The
stratified runs above cost 90-103 us per decode, all of the excess in the
conditional sampler, whose sequential (Chen-Liu) draw took one Python
step per mechanism, a fixed 13 ms per call on BB72's 2232 mechanisms that
the per-group draws of a few hundred to a few thousand shots turned into
30-145 us per shot. The sampler now draws positions in increasing order by
inverse transform on one cumulative table per remaining count (same law,
tested against the sequential scheme), and syndromes are formed by
gathering the fired mechanisms' columns and XOR-reducing per row. Measured
on the same machine: sampling 1 us per shot at every batch size, syndromes
4-7 us, Relay-BP decode 14-22 us, 19-28 us in total against 49-168 us
before; the same ten-repeat job fell from 37 min to 11 min and a
200k-budget run from 77.6 s to 31.8 s. The wall-time ratio against direct
Monte Carlo therefore equals the decoder-call ratio of section 4.

## 6. The certification allowance

The allowance epsilon is the one tuning knob of LACE. Too tight and the
never-failing groups eat the budget (44% of the Relay-BP decodes at
epsilon 0.15, and the certification floors overrun the nominal budget,
2.5x at a 200k budget); too loose and the bound dominates the certified
variance. `optimal_certify_fraction` (`--certify-fraction auto`) tries to
remove the knob: each round it minimises A / (N - N_cert(epsilon)) +
epsilon f / N over epsilon, where A is the Neyman variance constant from
the shrunk rates and N_cert the labels the floor rule needs, keeping the
failing groups at least a quarter of the budget. The realised bound
matched the targeted allowance within 10% in every run, so the rule does
what it says; the question is whether what it says is the right
objective. Relay-BP k = 3, seed 20261102 (20261101 for the 600k pairs),
ordered by realised decodes per arm so that neighbouring rows compare at
equal spend:

| allowance | nominal budget | decodes per arm | plug-in gain | certified gain | repeats certified below 1 | never-failing mass |
|---|---|---|---|---|---|---|
| auto (chose 0.5-2.5) | 200k | 317k | 2.43 | 0.59 | 30 of 30 | 0.46 |
| fixed 0.15 | 120k | 375k | 2.17 | **1.00** | 8 of 30 | 0.42 |
| fixed 0.15 | 200k | 513k | 2.72 | **1.24** | 15 of 100 | 0.29 |
| auto (chose 0.3-0.5) | 330k | 535k | 2.92 | 0.96 | 16 of 30 | 0.45 |
| auto (chose 0.25-0.5) | 600k | 948k | 3.49 | 1.43 | 0 of 10 | 0.72 |
| fixed 0.15 | 600k | 1.08M | 2.95 | **1.49** | 0 of 10 | 0.59 |

(`results/layers_bb72_relay_position_inround_{auto_200k_30rep,eps015_120k_30rep,eps015_coverage,auto_330k_30rep,auto_10rep,eps015_10rep_v2}.json`.)

At equal spend the auto rule realises 10-20% more plug-in gain and
certifies 20-40% less, with half or all of its repeats below 1 where the
fixed rule has 15-27%; only at about 1M decodes per arm do the two tie
on the certified figure. The reason is in the last column: the fixed rule
leaves 29-42% of the stratum mass never failing, the auto rule 45-72%.
The rule's projection treats the set of never-failing groups as given and
values certification labels only for shrinking their bound, whereas those
labels also turn near-empty groups into measured ones, which removes the
bound on them altogether; the rule undervalues that and so loosens the
allowance too far. Since the certified figure is what one run can
promise, the fixed allowance of 0.15 stays the default (it is also the
setting behind the coverage test and both ten-repeat runs); `auto` is
kept as the option that respects the nominal budget and maximises the
realised gain when a guarantee is not needed. Repairing the projection to
credit the discovery of failures is the obvious next improvement to the
rule and was not attempted.

## 7. What was tried and dropped

Documented briefly so it is not repeated; the reports were removed from
this branch.

* **Spatial (tile) refinement of the time statistic.** On surface d = 5
  with PyMatching the best spacetime statistic adds at most 6% to position
  + core at any count; on d = 7 it adds 8-31% bias-corrected at 2.4-3.3x
  the number of groups and shrinks as failures accumulate. Not a lever.
* **A cheap decoder as a control variate for an expensive one.** On BB72
  the correlation of BP-OSD failure with truncated BP (10-20 iterations)
  is 0.02-0.09 and with GPU Relay-BP 0.27-0.34 (Relay-BP misses 40-50% of
  BP-OSD's failures and fails 3x more often at k = 3); the ceiling with a
  free cheap arm is 1.13x. BP-OSD failures are mostly BP converging to a
  wrong codeword (only 15-30% had a non-converged BP), so the in-decoder
  flag is uninformative too. Decoders do not proxy each other.
* **Metropolis splitting (IBM technique III)** agreed with the spectrum
  down to p = 5e-4 and ran 25% low at 2.2e-4 with rising R-hat; the
  between-set spread does not cover a chain stuck in one failing sector.
* **A learned labelling or proposal** did not pay off against the exact
  stratification and adds a certification burden.

## 8. Conclusion

Exact fault-count stratification with exact reweighting is the unbiased
core of IBM's method and already beats direct Monte Carlo by two to three
orders of magnitude at LER ~ 1e-7; IBM's ansatz on top of it is biased by
up to 2x on BB72 and should not be used. The layer-stratified estimator
with certified allocation keeps that core, adds no model, and is verified
unbiased against an independent direct control at every rate, with
calibrated plug-in intervals and conservative certified ones. Against the
unbiased IBM baseline it is worth 1.4-1.6x certified and 2.5-3.5x
(Relay-BP) to 7-8x (BP-OSD) realised at the rates where the onset stratum
dominates, 930-2000x against direct Monte Carlo at p = 1e-4 in both
decoder calls and wall time, and about 1.03x for matching on the surface
code, where count strata alone suffice. Its one tuning knob, the
certification allowance, is best left at the validated fixed value; the
data-driven rule trades certified for realised gain. The natural next
study is BB144 and other qLDPC codes with BP-family decoders, where the
value concentrates.
