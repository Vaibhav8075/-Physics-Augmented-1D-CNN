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

## Addendum (later on 2026-10-17): sensitivity check and two added models

6. **Band-limit sensitivity check (new Section "Sensitivity to the Band Limit", Table "tab:sens").**
   P1 and P2 were repeated with a sharp elliptic low-pass (order 11, flat to 3.9 kHz, at least 120 dB
   down from 4.0 kHz) instead of the 3.6 kHz band limit, so the fault windows keep their 3.6-4 kHz
   content. Every mean difference keeps its sign and every conclusion holds; two significance calls near
   the threshold change (PAC-1DCNN's lead at 5 dB in P1 becomes significant; the white-noise loss of
   CNN+HP in the mixed regime does not remain so). Results: `results/sensitivity_sharp/`.

7. **Two models added (tertiary family, 28 tests per protocol and regime), both compared with WDCNN-s.**
   An ImageNet-pre-trained ResNet-18 on spectrograms (the network Hendriks et al. (2022) found strongest
   under their fault-size split) and WDCNN-s with AdaBN (batch-normalisation statistics re-estimated on
   the unlabelled test windows of each condition, as in the original WDCNN). The ResNet is never
   significantly better than WDCNN-s on CWRU and significantly worse in 9 of 28 P2 conditions, all at
   0 or -5 dB. AdaBN is significantly better at -5 dB under every noise spectrum in P2, P3 and P4 and
   significantly worse nowhere, while its clean scores do not change significantly. The abstract,
   model descriptions, statistics, results, discussion, threats and conclusion were updated; the model
   count is now twelve and the abstract stays within 250 words.

8. **Ranges now cover every network.** "The networks reach ..." ranges include the added models (P4
   clean macro-F1 now 36.7-45.0%, at most 12 points above random guessing; P2 fault-class F1 28.7-37.0%
   in the mixed regime). Kin-RF is the best model at 0 and -5 dB in every P3 condition, but in P4
   WDCNN-s+AdaBN is higher in 10 of 12; the discussion was corrected accordingly.
