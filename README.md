# Few-Shot Signer Style Transfer for Sign Language Recognition
### Learning an unseen signer's style from five clips to improve signer-independent recognition

**Notebook:** `Few_Shot_Signer_Style_Generation.ipynb` (Google Colab, T4 GPU)
**Data:** WLASL (American Sign Language), MediaPipe landmarks

---

## 1. Idea

My thesis separated *which character* is written from *whose handwriting style* it is, and generated new characters from five examples. This project applies the same idea to sign language: separate **which sign** is performed from **who performs it**.

**Research question.** If a sign language recognizer has never seen a signer, can a few-shot generative model learn that signer's style from only 5 clips, create extra training samples in that style, and improve recognition for that signer?

## 2. Method

1. **Data.** MediaPipe landmarks of WLASL videos (Kaggle dataset `abd0kamel/mutemotion-output`, downloaded with `kagglehub`, no login) plus the official `WLASL_v0.3.json` (gloss and signer id per video). Target vocabulary: WLASL100.
2. **Preprocessing.** Each clip is cut to the annotated sign, resampled to 32 frames, normalised for position and scale (movement style is kept), and reduced with PCA (fitted on training signers only).
3. **Signer-independent split.** A group of signers is held out completely. For each held-out signer, **5 clips are the support set** (few-shot examples) and the remaining clips are test clips.
4. **Style-conditioned VAE (trained on the other signers).**
   - *Style encoder:* a bidirectional GRU embeds each support clip; the style vector is the **mean** of the 5 embeddings (same design as my thesis FS-CVAE).
   - *Content:* a learned gloss embedding.
   - *Decoder:* a GRU that generates a 32-frame landmark sequence from a latent code, the gloss and the style.
   - *Discriminator:* a small GRU adds adversarial supervision for more realistic motion (switched on after a warm-up).
   - *Episodic training:* each episode picks one training signer, uses 5 of their clips as support and reconstructs a different clip of the same signer. Because the support clips show other signs, the style vector has to capture *how* the person signs, not *what* they sign.
   - Loss: reconstruction + velocity (motion) loss + KL (annealed) + adversarial.
5. **Style check.** For each unseen signer, predict their test clips from the gloss and a style vector only, once with the signer's own style and once with another unseen signer's style.
6. **Recognition experiment.** A bidirectional GRU recognizer is trained under four conditions (3 seeds each) and tested on the unseen signers:

| Condition | Training data |
|---|---|
| **A** | Real clips of training signers |
| **B** | A + the 5 support clips of each unseen signer |
| **C** | B + synthetic clips in the **target** signer's style |
| **D** | B + the same number of synthetic clips in a **random** signer's style (control) |

**The key comparison is C vs. D.** Both add the same number of synthetic clips, so if C is higher, the gain comes from the signer's style and not just from having more data. A paired bootstrap gives a 95 % CI for C−D and C−B.

## 3. Setup actually used in the saved run

| Item | Value |
|---|---|
| Frames per clip | 32 |
| PCA dimension | 96 |
| Support clips per signer (K) | 5 |
| Generator training steps | 4000 |
| Synthetic clips per gloss per test signer | 2 |
| Recognizer epochs / seeds | 30 / 3 |
| Clips after filtering | 641 |
| Glosses present | 39 |
| Signers | 56 (5 held out) |
| Train / support / test clips | 384 / 25 / 232 |

## 4. Results (from the saved run)

**Style check (own vs. other style).** Mean prediction error was 1.8723 with the signer's own style and 1.8729 with another signer's style. The own style was better for 80 % of the 5 unseen signers, but the average difference is tiny, so this is only weak evidence that the style vector carries signer information.

**Recognition on unseen signers (mean over 3 seeds):**

| Condition | Train clips | Top-1 % | Top-1 SD | Top-5 % | Rare-sign top-1 % |
|---|---|---|---|---|---|
| A: real | 384 | 37.93 | 1.86 | 66.24 | 19.34 |
| B: + 5 support clips | 409 | 37.64 | 1.42 | 68.10 | 19.75 |
| C: + target-style synthetic | 799 | 39.80 | 0.81 | 65.23 | 19.75 |
| D: + random-style synthetic | 799 | 37.64 | 0.89 | 64.66 | 17.70 |

C is about 2.2 points above D and B in top-1, which is the direction the hypothesis predicts. The paired-bootstrap confidence intervals are printed by the notebook (Step 6); check them before claiming any significant gain. Top-5 does not show the same benefit.

## 5. Honest reading and limitations

- **The effect is small.** The gain is about 2 points with only 5 test signers, 232 test clips and 3 seeds. Treat it as a promising pilot, not a proven improvement.
- **Reduced vocabulary.** Only 641 clips and **39 of the 100 glosses** had usable landmarks in the chosen file, so this is not a full WLASL100 result. `N_CLASSES` is 39 in practice.
- **Weak style signal.** The own-vs-other style difference in Step 4 is almost zero on average.
- One dataset (American Sign Language), landmarks only (no RGB), one recognizer architecture.
- Absolute accuracy is low because the training set is small and the split is signer-independent.

## 6. How to run

1. Open the notebook in Google Colab and select **Runtime → GPU (T4)**.
2. Run all cells in order. The landmark download is about 13 GB and takes most of the time (roughly 30 to 60 minutes in total).
3. Main settings are in the "Settings" cell (`N_GLOSSES`, `K_SHOT`, `GEN_STEPS`, `GEN_PER_GLOSS`, `CLS_EPOCHS`, `N_SEEDS`).

## 7. Outputs

Saved to `/content/signer_style/` and zipped as `signer_style_results.zip`:

- `style_check.csv`: own-style vs. other-style error per unseen signer
- `recognition_results.csv`: accuracy for conditions A to D
- `style_cvae.pt`: trained generator, glosses, PCA and normalisation parameters, test signer list
- plots generated by the notebook: `real_vs_generated.png`, `recognition.png`

## 8. Next steps

1. Use the Isharah dataset and Arabic Sign Language, where signers with very few clips (the singleton problem) make few-shot style transfer more relevant.
2. Use all 100 glosses (check why only 39 had usable landmarks) and more held-out signers, or leave-one-signer-out cross-validation.
3. Strengthen the style vector (for example a contrastive signer loss) and re-run the style check.
4. Try a stronger recognizer and report significance over more seeds.

## 9. Requirements

Python 3, PyTorch, NumPy, pandas, matplotlib, tqdm, `kagglehub`. All except `kagglehub` are pre-installed on Colab.

## 10. Data credit

WLASL: Li et al., *Word-level Deep Sign Language Recognition from Video* (WACV 2020). MediaPipe landmarks from the Kaggle dataset *MuteMotion: WLASL MediaPipe Encoded*. Please follow the terms of both datasets.
