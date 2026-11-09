# Few-Shot Signer Style Transfer for Sign Language Recognition
### Learning an unseen signer's style from five clips to improve signer-independent recognition

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


## 10. Data credit

WLASL: Li et al., *Word-level Deep Sign Language Recognition from Video* (WACV 2020). MediaPipe landmarks from the Kaggle dataset *MuteMotion: WLASL MediaPipe Encoded*. Please follow the terms of both datasets.
