# BioGuard-X v4 Dataset Guide

## Recommended layout

```text
participant_001/
  metadata.json
  face/
    session_1/*.jpg      # enrollment/reference
    session_2/*.jpg      # validation
    session_3/*.jpg      # final test
  fingerprint/
    session_1/*.png
    session_2/*.png
    session_3/*.png
  voice/
    session_1/*.wav
    session_2/*.wav
    session_3/*.wav
```

## Recommended collection

- Face: 5–10 good images per session, with different distance/lighting/pose but no extreme pose.
- Fingerprint: at least 2–5 captures per session of the same finger; preferably use the same sensor and known DPI.
- Voice: at least 2–5 recordings per session, around 2–8 seconds each, with different sentences and realistic room/microphone conditions.
- Use separate sessions on different days when possible.
- Reject blurry, clipped, silent, overexposed, or otherwise poor samples before enrollment.

## Evaluation protocol

`session_1` is the reference/enrollment set.

`session_2` is the validation set used to choose:

- face threshold
- fingerprint threshold
- voice threshold
- quality gates
- fusion weights
- fusion threshold

`session_3` is the untouched final test set.

Never tune a threshold on session 3 and then report the same session as an unbiased test.

## Metrics

Report each modality separately and then the multimodal system:

- FMR / FAR
- FNMR / FRR
- EER
- ROC-AUC
- confusion matrix
- confidence intervals
- latency
- quality rejection rate

## Privacy

Obtain informed consent before storing biometric samples. Keep the dataset local, restrict access, and delete a participant's samples when requested.
