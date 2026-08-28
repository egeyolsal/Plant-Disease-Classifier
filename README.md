# Plant-Disease-Classifier

An on-device plant species and disease classifier for Android. The app (**PlantInsight**) takes a photo of a plant leaf, identifies the species and health condition using a locally-run TensorFlow Lite model, compares the result against the PlantNet API, and shows care recommendations for a subset of classes.

The project's central finding: a model trained purely on lab-condition images (PlantVillage) loses most of its accuracy on real-world photos (PlantDoc), and this gap can be partially — but not fully — closed through domain adaptation. This document covers both how to run the project and the full technical narrative: every decision, every experiment that failed, and why.

## Results (final model)

Measured on a locked, held-out PlantDoc test set (384 real-world images, never used in training or model selection):

| Metric | Value |
|---|---|
| Disease-level Top-1 accuracy | 56.25% |
| Disease-level Top-3 accuracy | 81.25% |
| Disease-level F1-macro | 0.538 |
| Species-level Top-1 accuracy | 77.08% |
| Species-level F1-macro | 0.719 |
| On-device model size (TFLite, dynamic-range quantized) | 1.08 MB |
| Lab-condition (PlantVillage) Top-1, for comparison | 96.42% |

The 40-point gap between lab accuracy (96.42%) and real-world Top-1 (56.25%) is the project's core finding — see [The domain gap](#4-the-domain-gap-lab-vs-real-world) below.

## Table of contents

- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Running the mobile app](#running-the-mobile-app)
- [1. Dataset](#1-dataset)
- [2. Model architecture and baseline training](#2-model-architecture-and-baseline-training)
- [3. A methodology bug and its fix (Keras 3 evaluation)](#3-a-methodology-bug-and-its-fix-keras-3-evaluation)
- [4. The domain gap: lab vs. real world](#4-the-domain-gap-lab-vs-real-world)
- [5. Domain adaptation](#5-domain-adaptation)
- [6. A second bug: macro-F1 miscalculation](#6-a-second-bug-macro-f1-miscalculation)
- [7. Rejected improvement attempts](#7-rejected-improvement-attempts)
- [8. Rejected external datasets](#8-rejected-external-datasets)
- [9. Species-level evaluation](#9-species-level-evaluation)
- [10. Explainability (Grad-CAM)](#10-explainability-grad-cam)
- [11. Mobile export: TFLite and quantization](#11-mobile-export-tflite-and-quantization)
- [12. Mobile app development](#12-mobile-app-development)
- [13. Confidence threshold](#13-confidence-threshold)
- [14. PlantNet comparison](#14-plantnet-comparison)
- [15. Care recommendations](#15-care-recommendations)
- [16. Robustness testing](#16-robustness-testing)
- [17. UI design pass](#17-ui-design-pass)
- [Known limitations](#known-limitations)

## Project structure

```
Plant-Disease-Classifier/
├── configs/                      # class mapping, splits metadata, preprocessing spec,
│                                   confidence threshold decision, class weights
├── ml/
│   ├── src/
│   │   ├── data/                  # PlantVillage download, manifest, split, class weights
│   │   ├── preprocessing/         # augmentation preview
│   │   └── plantdoc/              # PlantDoc inspection, cropping, filtering, mapping
│   └── models/                    # final .tflite models (tracked); .keras checkpoints (not tracked)
├── mobile/                        # Flutter app (PlantInsight)
│   ├── lib/
│   │   ├── features/
│   │   │   ├── scan/                # camera/gallery input, result screen
│   │   │   ├── inference/           # TFLite classifier
│   │   │   ├── plantnet/            # PlantNet API comparison
│   │   │   └── care/                # care recommendations
│   │   └── core/theme/              # design system
│   └── assets/
│       ├── models/                  # bundled .tflite model + labels.json
│       └── data/                    # care_recommendations.json
├── notebooks/                     # Colab notebook: training, domain adaptation, TFLite export
├── reports/                       # every measured metric, as JSON/CSV/TXT, nothing invented
├── manifest_split.csv             # PlantVillage leak-safe split (leaf_id-level), one row per image
├── plantdoc_manifest.csv          # PlantDoc crop-level manifest (bbox source, mapped class, split)
├── requirements.txt
└── .gitignore
```

## Requirements

**For the ML pipeline:**
- Python 3.11+
- See `requirements.txt` for pinned package versions (TensorFlow 2.21.0, Keras 3.15.1, Hugging Face `datasets`, pandas, scikit-learn, Pillow)
- A GPU (via Google Colab or similar) is strongly recommended for training/domain-adaptation; data preparation runs fine on CPU

**For the mobile app:**
- Flutter 3.47.1+, Dart 3.13.1+
- Android Studio with Android SDK, NDK, and an emulator or physical device (minSdk 26, required by `tflite_flutter`)
- A free PlantNet API key from [my.plantnet.org](https://my.plantnet.org) (optional — the app works without it, PlantNet comparison simply shows an error)

## Running the mobile app

```bash
cd mobile
flutter pub get
flutter run -d <device-id> --dart-define=PLANTNET_API_KEY=<your-key>
```

Without `PLANTNET_API_KEY`, the app runs normally; the PlantNet card shows "PlantNet API key not configured" instead of a result.

---

## 1. Dataset

### PlantVillage (lab conditions)

The base dataset, loaded via Hugging Face `datasets`. Its original scripted loader is no longer supported by `datasets>=5.0` (script-based loading was removed), so this project downloads the raw files directly:

```bash
python ml/src/data/download_dataset.py
python ml/src/data/build_manifest.py
python ml/src/data/split_dataset.py
python ml/src/data/compute_class_weights.py
python ml/src/data/build_class_mapping.py
```

Verified counts: **54,305 images, 38 raw classes, 14 species.**

**Split methodology:** a naive random split risks *leakage* — near-duplicate photos of the same physical leaf (common in PlantVillage) ending up in both train and test. The split is grouped by `leaf_id` instead of by image, at the leaf level, 70/15/15: **37,950 train / 8,087 val / 8,268 test**, with a leakage check confirming zero overlap between splits. `leaf_id` coverage was 20,015 unique groups; 24.3% of images fell back to singleton groups (no match found in the source leaf-map), which carries no leakage risk since singletons can't repeat across splits.

Class imbalance is severe — roughly 35x between the largest and smallest class (`Orange___Haunglongbing` ~5,300 images vs `Potato___healthy` ~152). Class weights were computed on the train split only and used throughout training.

**Taxonomy decision:** only 5 species (Tomato, Potato, Apple, Grape, Corn) keep full disease-level detail in the target taxonomy (34 classes). The remaining 9 species collapse their disease/healthy sub-classes into a single species-level label (e.g. all Strawberry sub-classes become just `Strawberry`). This was a deliberate scope decision to avoid diluting training signal across too many low-data classes — it means, for those 9 species, the model cannot and was never trained to distinguish "healthy" from "diseased."

### PlantDoc (real-world conditions)

Downloaded from the official source — [Roboflow, joseph-nelson/plantdoc](https://universe.roboflow.com/joseph-nelson/plantdoc), version 4 (raw, no preprocessing/augmentation), COCO JSON format. Verified: 2,569 images, 13 species, 31 categories, 8,851 annotations. (Multiple unofficial/community re-uploads of PlantDoc exist on Roboflow with different class counts — this specific version was confirmed to match the metrics in the original PlantDoc paper.)

Bounding box format was verified empirically rather than assumed: `[x, y, width, height]` in absolute pixels, confirmed by cross-checking `area = width × height` against the recorded area field.

Processing pipeline:
```bash
python ml/src/plantdoc/inspect_plantdoc.py
python ml/src/plantdoc/crop_plantdoc.py
python ml/src/plantdoc/filter_small_crops.py
python ml/src/plantdoc/finalize_plantdoc_manifest.py
```

- One crop was produced per annotated bounding box: **7,931 crops** initially.
- **Quality control caught a real issue:** a sampled `Grape___Black_rot` crop was only 91×88px and visibly pixelated. Rather than assume the whole class was low-quality from one example, per-class size statistics were computed — the class's *median* size was actually 273px; the sampled image was an outlier, not representative. This is a concrete instance of the project's rule to verify with data rather than a single glance.
- Filtering removed crops below 50px on their shortest side: **584 crops (7.4%)** were dropped. `Tomato___Tomato_mosaic_virus` was disproportionately affected (median crop size only 65px before filtering).
- `Tomato___Spider_mites_Two-spotted_spider_mite` was excluded entirely — only 2 crops remained after filtering, too few for any meaningful training or evaluation.
- **Final PlantDoc dataset: 7,345 crops across 23 of the 34 target classes** (6,010 train / 951 valid / 384 test).
- **11 of the 34 target classes have zero PlantDoc coverage:** `Apple___Black_rot`, `Apple___healthy`, `Corn_(maize)___healthy`, `Grape___Esca_(Black_Measles)`, `Grape___Leaf_blight_(Isariopsis_Leaf_Spot)`, `Grape___healthy`, `Orange`, `Potato___healthy`, `Tomato___Target_Spot`, `Tomato___healthy`, plus `Tomato___Spider_mites...` (excluded for insufficient data as above). Their real-world accuracy is genuinely unmeasured — not assumed to be good or bad.

The PlantDoc test split (384 crops) is treated as a **hard rule**: it is used for final evaluation only and was never touched by any training, fine-tuning, or hyperparameter/threshold selection at any point in this project.

## 2. Model architecture and baseline training

Two architectures were compared, both via transfer learning (frozen ImageNet backbone, trained classification head) on PlantVillage:

| Model | Top-1 | F1-macro | Size (MB) | CPU inference (ms) |
|---|---|---|---|---|
| MobileNetV3Small (baseline) | 0.9490 | 0.9343 | 4.36 | 175.16 |
| EfficientNetB0 (baseline) | 0.9657 | 0.9583 | 16.77 | 324.05 |
| MobileNetV3Small (fine-tuned) — **final choice** | 0.9642 | 0.9548 | 9.24* | 213.48 |

\*Includes optimizer state; the exported TFLite model is much smaller (see [Section 11](#11-mobile-export-tflite-and-quantization)).

**Note on EfficientNet-Lite0:** the original plan called for EfficientNet-Lite0 (purpose-built for mobile/TFLite quantization), not EfficientNetB0. Two independent Lite0 implementations failed with the same root cause — `keras.utils.layer_utils` was removed in Keras 3 — so EfficientNetB0 was substituted, documented as a deliberate swap rather than a silent one.

**Fine-tuning:** the last 2 backbone blocks (layer indices 123–156) were unfrozen, BatchNorm layers were kept frozen (standard guidance for small/imbalanced batches), trained for 10 epochs at `lr=1e-5`. Verified trainable parameters: 658,114 across 26 unfrozen layers.

**Final architecture choice:** despite EfficientNetB0's higher lab accuracy (+1.7 Top-1 points), **MobileNetV3Small was selected** for the mobile deployment — it is ~3.8x smaller (4.36MB vs 16.77MB) and ~1.6x faster on the same CPU benchmark, which matters more for an on-device app than a small accuracy edge.

## 3. A methodology bug and its fix (Keras 3 evaluation)

Early in training, `EfficientNetB0` appeared to collapse to 1.68% test accuracy despite 96%+ training accuracy. Root cause: calling `model.predict(test_ds)` immediately after `model.fit()` — with `EarlyStopping(restore_best_weights=True)` — does not reliably reflect the restored best weights in Keras 3; the in-memory model object can be in a stale state.

**Fix applied everywhere from that point on:** always reload the model explicitly from its saved `.keras` checkpoint file before any evaluation, never trust the in-memory post-`fit()` object. This is now a standing rule throughout the project's notebook.

## 4. The domain gap: lab vs. real world

With the fine-tuned MobileNetV3Small (96.42% lab Top-1) evaluated on the **locked PlantDoc test set** (384 real-world images, 23 classes) with no adaptation:

| Metric | PlantVillage (lab) | PlantDoc (real-world, before adaptation) |
|---|---|---|
| Top-1 | 96.42% | **26.82%** |
| Top-3 | — | 47.40% |
| F1-macro | 0.9548 | **0.1681** |

Five classes collapsed entirely (F1 = 0): `Apple___Cedar_apple_rust`, `Grape___Black_rot`, `Tomato___Bacterial_spot`, `Tomato___Leaf_Mold`, `Tomato___Tomato_mosaic_virus`. `Tomato___Late_blight` showed severe over-prediction (recall 0.7143, precision 0.1351) — the model defaulted to this class whenever uncertain.

This ~70-point Top-1 drop is the project's central empirical finding: a classifier that looks excellent on lab-style images (clean background, isolated leaf, controlled lighting) can perform barely better than random on real-world smartphone photos.

## 5. Domain adaptation

**Strategy:** fine-tune on a mix of PlantVillage (lab) train data and PlantDoc (real-world) train data, using PlantDoc's own validation split — not PlantVillage's — as the early-stopping signal, so model selection is driven by real-world performance.

Mixed set: 37,950 (PlantVillage) + 6,010 (PlantDoc train) = 43,960 images, a natural ~6.3:1 ratio (no artificial oversampling of PlantDoc was applied for the primary run).

**Standard fine-tuning**, run to 24 epochs total, showed steady (if noisy — 5 of the last 15 epochs had a small regression, but gains consistently outweighed them) improvement:

| Stage | Top-1 | Top-3 | F1-macro* |
|---|---|---|---|
| Before adaptation | 0.2682 | 0.4740 | 0.1681 |
| +10 epochs | 0.4583 | 0.7682 | 0.3707 |
| +24 epochs | 0.5417 | 0.7995 | 0.4734 |

\*Later found to be understated due to a calculation bug — see [Section 6](#6-a-second-bug-macro-f1-miscalculation).

Per-class analysis found only a **weak correlation (0.373)** between a class's PlantDoc image count and its resulting F1 — most weak classes were not simply data-starved. Four of five previously-collapsed classes recovered; `Tomato___Bacterial_spot` did not, despite 227 training images — see [Section 10](#10-explainability-grad-cam).

**Focal loss** (γ=2, α=0.25) was tried next, since standard fine-tuning still left several classes under-served. It reached a best checkpoint at epoch 8 (F1-macro 0.4941) before continuing to an early-stopped epoch 9 (Top-1 0.5625, Top-3 0.8125, F1-macro 0.4759, figures affected by the same bug as above) — **this became the final model, `mobilenetv3small_focal.keras`.**

**A real mistake here, documented rather than hidden:** the epoch-8 checkpoint (a higher F1-macro than the final epoch-9 result) was silently overwritten by `ModelCheckpoint` before being backed up — `save_best_only=True` restarts its "best so far" tracking from `None` on every new `model.fit()` call, rather than remembering a previous run's true best. It could not be recovered. Standing rule adopted afterward: **back up a checkpoint before resuming training past it.**

## 6. A second bug: macro-F1 miscalculation

While evaluating whether to drop the worst-performing class (`Tomato___Bacterial_spot`) from the taxonomy, a sanity check surfaced a serious methodology bug: **removing the single worst-performing class from a macro-average should mathematically raise or hold the F1-macro score, never lower it.** An initial (flawed) measurement showed a drop (0.4759 → 0.3776), which was the tell that something was wrong.

**Root cause:** every `f1_score(..., average='macro')` call from Day 6 onward had been made **without an explicit `labels=` parameter**. Since 11 target classes never appear in PlantDoc's ground truth (`y_true`, no test data for them) but the model occasionally predicts them anyway (they appear in `y_pred`), scikit-learn silently included these as phantom F1=0 "classes" in the macro average — deflating every previously reported PlantDoc F1-macro score across the entire project.

**Corrected value for the final model**, counting only the 23 classes with actual PlantDoc test data: **F1-macro = 0.5380** (not the 0.4759 reported at the time). Top-1/Top-3 were unaffected, since accuracy doesn't depend on per-class averaging. `precision_macro`/`recall_macro` from the same pre-fix calls were not independently re-verified and are not reported anywhere in this document as a result — only the corrected, verified metrics (Top-1, Top-3, F1-macro) are used.

This does not change any of the *relative* conclusions already drawn (e.g., which classes recovered under focal loss, which experiments helped or hurt) since those comparisons used the same consistent — if buggy — method throughout. It does mean the final model performs meaningfully better in absolute terms than was believed for several days of the project.

## 7. Rejected improvement attempts

Three further ideas were tried and measured — not assumed to work — after focal loss became the leading candidate.

**Per-class confidence calibration:** searching for a per-class softmax bias correction that would improve macro-F1 (optimized on validation only). The optimizer converged to **zero adjustment for every class** — no improvement found, abandoned.

**Targeted augmentation for `Tomato___Tomato_mosaic_virus`** (genuinely data-starved at 80 PlantDoc images, unlike Bacterial_spot's data-quality problem): 4x-oversampling with heavier augmentation, touching only this one class. Its F1 improved substantially (0.2791 → 0.4000, later confirmed live — [Section 10](#10-explainability-grad-cam)), but **overall F1-macro dropped slightly** (0.4759 → 0.4655). Rejected as the final model since it hurt the metric the project prioritized.

**Scope reduction — excluding `Tomato___Bacterial_spot` entirely:** given its diagnosed, unfixable data-quality problem ([Section 10](#10-explainability-grad-cam)), a genuine 33-class model was retrained from scratch (verified: exactly one class removed, nothing else touched). Result: **F1-macro 0.5298, Top-1 0.5514 — both worse than the 34-class model.** A per-class comparison showed this wasn't a systematic effect of removing Bacterial_spot (the classes that shifted most — Cherry, Soybean, Tomato_Late_blight — have no relationship to it), consistent with ordinary training-run variance since no random seed was fixed. **Rejected** — the 34-class model remains final, which also means the product supports 34 diagnoses rather than 33.

## 8. Rejected external datasets

In pursuit of improving the weakest classes, eight external dataset candidates were investigated and rejected — each for a genuine, verified reason:

| Candidate | Verdict | Reason |
|---|---|---|
| Plant Pathology 2021 (FGVC8) | Rejected | Apple-only; no overlap with the actual weak classes |
| Cassava Leaf Disease | Rejected | Species mismatch; "Cassava Mosaic Disease" only shares a name with "Tomato mosaic virus," not a pathogen family or host |
| TOM2024 (Burkina Faso) | Rejected | Closest by crop coverage (maize/tomato), but its `virosis_d` category — which looked like a name-level match for mosaic virus — showed filiform/shoestring leaf deformation on visual inspection, not classic mosaic mottling; a name match that failed a visual check |
| Mendeley "Tomato Leaf Disease Dataset" | Rejected | Class names use PlantVillage's exact `___` convention; every class has precisely 1,100 images (statistically implausible for field-collected data); the dataset's own published notebooks show train/val/test drawn from the same 11,000-image pool with no independent holdout, yielding an implausible F1=1.00 on mosaic virus |
| Multi-Crop Disease Dataset | Rejected | Covers banana, pepper, radish, peanut, cauliflower — irrelevant to the weak classes |
| Kaggle "corn-leaf-diseases-dataset" | Rejected | Its own description states it was "made from a bigger dataset" — likely a repackaged subset of PlantVillage/PlantDoc |
| GitHub `plant_village` mirror repo | Rejected | Directly credits the same PlantVillage authors and reports the same ~54,300 image count — a repackaging, not new data |
| Mendeley Corn dataset (`hmkd6nbngr`) | Rejected | Cross-referenced against an academic paper with identical per-class image counts (4,188 total; 1,146/1,306/574/1,162 per class) and a GitHub repo explicitly stating it was "made using PlantVillage and PlantDoc" — confirmed repackaging, not independent data. Also visually: tarla-photo and studio-background images were mixed within the same category folder, matching the same source-inconsistency pattern found in PlantDoc's own Bacterial_spot category |

The pattern across all eight is itself a finding: **apparent name or topic matches require verification before integration** — repackaged copies of data already in hand add no value, no matter how large they claim to be.

## 9. Species-level evaluation

Disease-level accuracy (Section 4–7) is a strict, exact-match metric — getting the species right but the specific disease wrong still counts as a full miss. A second, legitimate view collapses each predicted/true class to its species prefix (everything before `___`):

| Metric | Value |
|---|---|
| Species-level Top-1 | 77.08% |
| Species-level F1-macro | 0.719 |
| (for comparison) Disease-level Top-1 | 56.25% |
| (for comparison) Disease-level F1-macro | 0.538 |

Verified before reporting: the sets of predicted and true species match exactly (13/13), confirming no parsing artifacts from complex class names containing parentheses, commas, or spaces (e.g. `Corn_(maize)`, `Pepper,_bell`, `Cherry_(including_sour)`).

Both metrics are reported together in the app and in this document — they measure genuinely different, legitimate capabilities (recognizing the plant vs. diagnosing the specific condition), not a way to present a single inflated number.

## 10. Explainability (Grad-CAM)

Grad-CAM was implemented against the backbone's `activation_17` layer (last post-ReLU activation, 7×7×576 spatial output), verified by inspecting the actual layer graph rather than assuming a name. A two-stage sub-model (backbone output → remaining classifier layers) was needed to work around a Keras 3 limitation with nested `Functional` models.

**Root-cause diagnosis of `Tomato___Bacterial_spot`** (F1 ≈ 0.09 across every strategy tried): visual inspection of all 14 PlantDoc test images found they are **not a coherent dataset** — scanned publication figures, studio-background posters, and real phone photos mixed under one label, with at least one image showing symptoms inconsistent with bacterial spot. **This is a source data-quality problem, not a data-quantity problem** — no amount of additional training or external data ([Section 8](#8-rejected-external-datasets)) can fix a category whose own images don't agree on what the disease looks like.

Other samples confirmed the model working as expected: `Corn_(maize)___Common_rust_` (88%, heatmap tight on the actual rust pustules) and `Tomato___Tomato_mosaic_virus` (75%, correct — live confirmation that the targeted augmentation from Section 7 improved a real prediction, not just the aggregate number). One outlier was traced rather than assumed representative: a `Strawberry` image misclassified as `Raspberry` turned out to be a watermarked stock photo, not typical of that class's actual strong performance (F1 = 0.91).

## 11. Mobile export: TFLite and quantization

| Format | Top-1 | F1-macro | Size |
|---|---|---|---|
| Keras (float32) | 56.25% | 0.5380 | 3.63 MB |
| TFLite, float32 | 56.25% (1.0000 prediction agreement with Keras, 30-sample check) | — | 3.63 MB |
| TFLite, **INT8 (full)** | — | — | 1.18 MB — **converted successfully but failed at inference time** |
| TFLite, **dynamic-range (final)** | 54.69% | 0.5266 | **1.08 MB** |

Full INT8 quantization converted without error but the resulting model **could not be loaded at inference time**: `RuntimeError: failed to create XNNPACK runtime, Node 119 (TfLiteXNNPackDelegate) failed to prepare`. This is most likely due to MobileNetV3Small's squeeze-and-excitation blocks not being fully INT8-compatible with the XNNPack delegate. Disabling XNNPack explicitly was attempted and did not resolve it. Rather than force a fragile fix, the project fell back to **dynamic-range quantization** (weights only quantized to INT8, activations remain float32) — more broadly compatible, verified to load and run correctly, and it produced the smallest file of the three formats tested (smaller than full INT8, likely because INT8 carries extra per-layer calibration metadata that outweighs its activation savings for a model this size).

The dynamic-range model — bundled into the Flutter app — costs about 1.6 accuracy points relative to the source Keras model, judged an acceptable trade for a ~70% size reduction (3.63MB → 1.08MB).

## 12. Mobile app development

**Environment setup surfaced two real build blockers**, both root-caused rather than worked around blindly:
- A non-ASCII character in the project path (Turkish "Masaüstü") broke Gradle's native build tooling, which explicitly warns against non-ASCII paths on Windows. Fixed by relocating the repository to an ASCII-only path.
- The default NDK version Flutter expected (`28.2.13676358`) was not installed, and the newest Android SDK command-line tooling (`sdkmanager` → `android` CLI transition) failed to install it via the command line with a confusing "package not found" error caused by its new argument parser mis-splitting the versioned package string. Resolved by installing a compatible NDK version via the Android Studio GUI (which was unaffected by the CLI regression) and pinning `ndkVersion` explicitly in `build.gradle.kts` rather than relying on Flutter's default.

**TFLite integration surfaced a deeper build issue:** `tflite_flutter`'s own `android/build.gradle` (plain Groovy) hard-codes Java 11, while this project (AGP 9.1.0, built-in Kotlin compilation) targets Java/Kotlin 17 throughout — producing an "Inconsistent JVM Target Compatibility" failure. Five Gradle configuration approaches failed in sequence (each hitting a different property-finalization timing issue specific to AGP 9.1's build lifecycle) before the working fix was found: configuring `JavaCompile`/`KotlinCompile` tasks directly inside `gradle.projectsEvaluated { }`, which runs after all subprojects are fully evaluated and so never attempts to write to an already-finalized property.

`minSdk` also had to be pinned to 26 (from Flutter's resolved default of 24), since `tflite_flutter` requires API 26+.

The classifier was verified live on an Android emulator (Pixel 8, API 36): a real PlantDoc test image (`Apple___Apple_scab`) produced a correct top-1 prediction at 38.0% confidence — consistent with, not contradicting, the measured real-world Top-1 accuracy of 56.25%.

**Preprocessing correctness:** the on-device pipeline passes raw `[0, 255]` float32 pixels with no manual rescaling, matching training exactly — MobileNetV3's `include_preprocessing=True` handles normalization internally on both the Python and TFLite sides. Getting this wrong (e.g. dividing by 255 on the mobile side) would silently produce plausible-looking but wrong predictions, so this was checked explicitly rather than assumed.

## 13. Confidence threshold

A threshold below which the app shows a "low confidence" warning was selected empirically on the PlantDoc **validation** set (951 images — never the test set), by measuring how predictive confidence actually correlates with correctness at several candidate cutoffs:

| Threshold | % of predictions flagged as uncertain | Accuracy *above* threshold | Accuracy *below* threshold |
|---|---|---|---|
| 0.30 | 9.1% | 60.88% | 20.69% |
| 0.40 | 20.7% | 64.99% | 27.41% |
| **0.45 (selected)** | **26.6%** | **66.76%** | **30.83%** |
| 0.50 | 32.2% | 68.84% | 32.68% |
| 0.60 | 45.7% | 73.26% | 38.16% |
| 0.70 | 57.7% | 80.10% | 40.44% |

0.45 was chosen over the initially-considered 0.5 specifically to reduce how often users see the warning (26.6% vs 32.2% of scans) while retaining nearly identical discriminative power (a 2.17x accuracy ratio between above/below-threshold predictions, vs. 2.11x at 0.5) — a deliberate UX trade-off made with the actual numbers in hand, not a guess.

Verified live on both branches: a real-world image at 38.0% confidence correctly triggered the warning; a lab-quality image of the same class at 97.6% confidence correctly did not.

## 14. PlantNet comparison

The app also queries the [PlantNet API](https://my.plantnet.org) (`organs=leaf` explicitly set) as an independent species-only reference, shown alongside — not instead of — the on-device result. The API key is read at build time via `--dart-define`, never hard-coded or committed.

**A real, informative disagreement was observed during testing:** on a diseased PlantDoc image (`Apple___Apple_scab`), PlantNet's top match was **wrong** (`Rosa spp.`, 36.0%, with `Malus spp.` — the correct genus — only second at 11.8%), while the on-device model correctly identified both species and disease with high confidence. A cleaner reference image later returned a correct PlantNet match (`Crab Apple`, a `Malus` species). This suggests PlantNet — a general-purpose plant identifier, not disease-specialized — also degrades on diseased or atypical leaf imagery, similar in kind to the domain gap problem this project spent most of its time addressing.

The UI explicitly states that PlantNet identifies species only, not disease, and that its scores are not directly comparable to the on-device model's — this is a stated design constraint, not an oversight.

## 15. Care recommendations

Static, per-class guidance shown when the top prediction has an available entry — currently **6 of 34 classes**, deliberately not all 34, since fabricating agricultural advice for classes without a reviewed source would be worse than showing nothing.

Selection was based on the final model's actual measured PlantDoc F1-score, not guesswork:
- **4 disease-specific classes** (highest real-world F1 among diagnosable diseases): `Corn_(maize)___Common_rust_` (0.78), `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (0.62), `Tomato___Early_blight` (0.58), `Corn_(maize)___Northern_Leaf_Blight` (0.57).
- **2 species-only classes** (highest F1 among the species-only category, to demonstrate general-care content as a distinct type from disease treatment): `Strawberry` (0.91), `Blueberry` (0.81).

Content was researched from university/extension sources (Purdue, Nebraska, Delaware, Cornell, SDSU, Clemson, Oregon State, Illinois, and Colorado State extension programs), not invented. Specific fungicide product names and dosages found in those sources were **deliberately excluded** — this kind of information is region- and regulation-dependent, can go stale, and could cause real harm if misapplied; entries instead point the user to a local extension office. The `Tomato___Tomato_Yellow_Leaf_Curl_Virus` entry states plainly that infected plants cannot be cured, since that is what the sources say — management is prevention/vector-control only. The two species-only entries explicitly state that no specific disease was diagnosed for that class, so as not to imply false precision.

Classes without a care entry show a plain "not available yet" notice rather than a missing button or a crash.

## 16. Robustness testing

Four scenarios were deliberately tested live, not assumed to work: no internet (local model keeps working, PlantNet fails gracefully), repeated consecutive scans (state resets cleanly, no stale cards), and camera permission denial/re-grant (handled without crashing).

The fourth scenario — **a completely unrelated image (a desktop screenshot)** — revealed a genuine capability gap rather than a bug: the local model classified it as `Strawberry` at **55.6% confidence**, above the 0.45 threshold, so no warning was shown despite the prediction being entirely wrong. PlantNet, by contrast, correctly rejected the same image with an HTTP 404 "Species not found" (confirmed via PlantNet's own FAQ to be deliberate no-plant-detected behavior). This project's classifier has no equivalent mechanism — it was only ever trained to choose among 34 known classes and has no way to express "none of the above."

Two real bugs were found and fixed along the way: `ImageInputService` was showing the raw `PlatformException.toString()` on permission denial (literally `PlatformException(..., null, null)`) instead of a real message — fixed to show only the exception's `.message` field. `PlantNetService` showed a generic "API error (404)" for what is actually PlantNet's documented "no plant detected" response — fixed to show an accurate message.

## 17. UI design pass

The final visual design pass was carried out under a scoped brief that explicitly forbade touching any `services/` file, state management logic, the `0.45` confidence threshold, any disclaimer/warning text, or asset paths — since those encode decisions validated throughout the rest of this document. Compliance was verified afterward via `git diff --stat`, confirming only the three intended files changed (`app_theme.dart`, `scan_screen.dart`, `care_screen.dart`) with no service, `main.dart`, or dependency changes.

The result: a full Material 3 design system (a deeper, higher-contrast green for outdoor legibility, a tuned surface-container ramp for card layering, a deliberately calm amber — not red — palette for the low-confidence notice so it reads as informative rather than as a failure state), a redesigned result card with an animated confidence bar and proportional prediction breakdown, and a visually de-emphasized PlantNet card to reinforce that it's a secondary, non-comparable reference (per Section 14). All disclaimer and warning text was verified unchanged, character for character, and all four robustness scenarios plus both confidence-threshold branches were re-verified live on the emulator after the redesign.

---

## Known limitations

- The classifier only recognizes **leaf** images; behavior on other plant parts (fruit, bark, whole-plant photos) is untested.
- The model has **no "not a plant" rejection mechanism** — an unrelated image will still be classified into one of the 34 known classes, sometimes with high confidence (observed: 55.6% on a clearly non-plant test image, above the warning threshold). PlantNet has this mechanism (its API returns 404 by design in that case); this project's model does not.
- **11 of the 34 target classes have no PlantDoc (real-world) test data at all**; their real-world accuracy is genuinely unmeasured, not assumed good or bad.
- `Tomato___Bacterial_spot` performs poorly (F1 ≈ 0.09) across every strategy tried (standard fine-tuning, focal loss, targeted augmentation). Root-cause analysis (Grad-CAM + manual image review) traced this to inconsistent, seemingly miscategorized source images within PlantDoc's own category for this class — not to insufficient training data, and not something more data (from any of the 8 external sources investigated) would fix.
- Care recommendations are available for only 6 of 34 classes; the rest show a "not available yet" notice rather than fabricated advice.
- No random seed was fixed during model training, so exact metric reproduction from a from-scratch retrain is not guaranteed (the 33-class scope-reduction experiment demonstrated real run-to-run variance of this kind).
- `precision_macro`/`recall_macro` for the final model were computed before the F1-macro bug fix (Section 6) and were not independently re-verified; they are not reported anywhere in this document for that reason, only Top-1/Top-3/F1-macro.
