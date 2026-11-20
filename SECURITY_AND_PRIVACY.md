# Security and Privacy — BioGuard-X v4

## Data handling
- Biometric files remain under `datasets/custom/` and are not uploaded by the application.
- Face embeddings are cached locally for the face gallery.
- Fingerprint and speaker templates are computed at runtime from enrolled local samples.
- User deletion removes the user's biometric folder and enrollment rows.
- The application requires informed consent before enrollment.

## Accuracy and security limits
- No biometric decision is guaranteed to be correct.
- Thresholds in `src/config.py` are starting values, not measured accuracy claims.
- Final thresholds must be calibrated on a validation split and evaluated on a separate test split.
- Liveness is a challenge-response control and must be evaluated against genuine and attack clips before being described as anti-spoofing protection.
- Fingerprint matching depends strongly on sensor DPI and capture quality.
- Speaker recognition depends on microphone/channel/noise conditions and should be evaluated on the intended deployment environment.

## Recommended dataset protocol
Use separate capture sessions per participant:

`session_1 = enrollment`, `session_2 = validation`, `session_3 = final test`.

Do not tune thresholds, fusion weights, or quality gates on the final test set.
