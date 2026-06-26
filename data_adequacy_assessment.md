# Data-Adequacy Assessment — v7 dataset (2026-06-15)

**Question:** is the data we've gathered enough for good results when real-world testing
on the printer becomes possible? (Real testing deferred — no printer access right now.)

**Method:** counted how each v7 class is actually sourced (genuine colour-diverse *real* data
vs synthetic composites vs gold-lab Kaggle), since the v6→v7 evidence is that **only
colour-diverse real data transfers to the real world** — synthetic/gold-lab data inflates the
validation score but fails on real prints (the Warping domain-gap).

## Per-class real-data backing

| Class | diverse-real | synthetic | gold-lab | Verdict |
|---|---|---|---|---|
| Spaghetti | 2010 | 150 | 0 | ✅ **Sufficient** — abundant real, tight-boxed |
| Stringing | 1576 | 300 | 0 | ✅ **Sufficient** |
| Cracking | 993 | 150 | 0 | ✅ **Sufficient** (diverse colours, no gold bias) |
| Warping | 30 | 400 | 0 | ⚠️ **Marginal** — first real data ever; thin; synthetic is gold-lab |
| Blob_of_death | 26 | 150 | 0 | ⚠️ **Marginal** — diverse but thin (new class, v1) |
| Layer_shifting | **0** | 400 | 150 | ❌ **Insufficient** — no diverse real data at all |

## Findings

1. **Tier 1 (ready): Spaghetti, Stringing, Cracking.** Hundreds–thousands of real, colour-diverse,
   tight-boxed images (Roboflow + scraped). These should transfer to real testing.

2. **Tier 2 (a thin but real foundation): Warping (30), Blob (26).** First real data for both.
   Enough to *start* and far better than v6 (warping was 0 real), but thin. Warping's 400 synthetic
   are gold-lab composites that may *reinforce* the Warping↔Layer_shifting confusion rather than help.

3. **Tier 3 (the critical gap): Layer_shifting = "the next Warping".** ZERO colour-diverse real data —
   100% synthetic + gold-lab Kaggle, structurally identical to the v6 warping setup that failed 0/3 on
   real images. Its 0.99 val AP and 3/3 holdout are gold-domain artefacts and will NOT predict real-world
   performance. Because Warping and Layer_shifting are mutually confusable, **both** need diverse real
   data for either to be reliable on real prints.

## Recommended data to gather BEFORE real testing (priority order)

1. **Layer_shifting — HIGHEST.** Scrape ~30–50 colour-diverse real layer-shift images (the scraper at
   `data/scripts/scraper/` supports adding a target; it was never run for layer_shifting). This is the
   single biggest risk to real-world results.
2. **Warping — expand 30 → ~60–80,** prioritising *wide / webcam-above-bed* views that match the Elegoo
   deployment camera (current 30 skew to side-view/macro). Consider reducing the gold synthetic share.
3. **Blob — expand 26 → ~50** when convenient (current set is a usable v1).
4. **When the printer is available (the real fix):** capture real examples of *every* class on the Elegoo
   OrangeStorm Giga through the actual deployment camera, and hold a portion out as a true test set. This
   is the gold-standard validation and is exactly what `src/datacollector.py` (active-learning) is meant to
   feed. It also gives the first *fair* Warping/Layer_shifting real-world numbers (the current holdout's
   warping/layer images are gold-lab and out-of-domain).

## Bottom line
Half the classes (Spaghetti/Stringing/Cracking) are data-ready for real testing. Warping and Blob have a
thin-but-real foundation. **Layer_shifting is not yet data-ready** and should be the next scrape. None of
this blocks writing the thesis — the synthetic-to-real gap and this tiered readiness are themselves a
strong, honest results/methodology story.
