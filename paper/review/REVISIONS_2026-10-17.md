# Revisions after re-running the CWRU experiments (2026-10-17)

Changes since the review draft of 2026-10-14. Unlike that revision, this one changes results: all CWRU
experiments (P1, P2, the demonstration model, fold-level statistics and the convergence check) were
re-run with corrected preprocessing. The Paderborn experiments (P3, P4) are unaffected and were not
re-run. Every number in the paper is still generated from the saved results.

1. **CWRU band limit lowered from 4.0 to 3.6 kHz.** Smith and Randall (2015, Sec. 6.1.1) report an
   interference line near 4.2 kHz in the CWRU data. Because the healthy vibration is weak, this line holds
   27-64% of a standardised healthy window's power but under 1% of a faulty one, so it marks the healthy
   class. The old 4.0 kHz band limit attenuated it by only 0-21 dB and it still separated the classes
   under the training noise. The 3.6 kHz band limit attenuates it by 47-73 dB; under the training noise
   its separability is now at most 0.53 (0.5 = no separation). The cost is the 3.6-4 kHz band, which
   holds about 12% (inner race), under 1% (ball) and 10% (outer race) of a fault window's power; the strongest
   fault content above 2 kHz (2.8-3.5 kHz) is kept. Section III-B reports this trade-off, and the open
   item about the line was removed from Threats to Validity and Future Work. Two regression tests check the
   filter response and the remaining line level.

2. **Changed conclusion, P1 (leave-one-load-out).** PAC-1DCNN is now significantly better than the
   standard CNN at 0 dB (74.8 vs 62.8% macro-F1, ahead in all 20 runs and all four folds). With the 4.0 kHz
   band limit the mean difference was similar but not significant after Holm correction. Without its
   front end PAC-1DCNN reaches 70.7%, not significantly different from the full model, so the advantage
   does not come from the physics front end. P1 shares all physical bearings between training and test.
   The abstract, introduction, P1 results, discussion and conclusion no longer say that PAC-1DCNN is
   never significantly better; they now say it is significantly better only on this split.

3. **Changed conclusion, P2 (fault-size hold-out).** The front end's gain for the standard CNN under
   low-pass noise at 0 dB (+6.5 and +9.0 points, 13 of 15 runs in each training regime) is no longer
   significant. Its loss under white noise at 0 dB is significant in both regimes (2 of 56 front-end
   comparisons). The paper now says the front end significantly lowers macro-F1 under white noise and
   *tends* to help under low-pass noise. The single-stream variant (PAC-dual) is significantly better under
   low-pass noise and now significantly worse under white noise, not high-pass noise. The WDCNN-style
   network is significantly better than PAC-1DCNN in 6 of 28 conditions (low-pass noise at 5, 0 and
   -5 dB). The main message is unchanged: the front end's effect follows the spectrum of the test noise,
   and with unseen defect sizes PAC-1DCNN is never significantly better than the standard CNN.

4. **Fault-class F1 added.** On CWRU P2 the single healthy bearing is split in time, so the healthy class
   leaks and inflates macro-F1. The paper now also reports the mean F1 over the three fault classes only:
   28.7-33.7% for the networks, against 48.5% (envelope-spectrum random forest) and 49.7% (kinematic random
   forest), clean data, mixed training regime.

5. **Abstract** rewritten for points 2 and 3 and kept within 250 words.
