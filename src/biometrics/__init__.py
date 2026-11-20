"""Lean multimodal biometric stack: face, fingerprint, voice, and score fusion."""
from .fingerprint import FingerprintMatcher
from .voice import SpeakerEncoder
from .fusion import MultimodalFusion

__all__ = ["FingerprintMatcher", "SpeakerEncoder", "MultimodalFusion"]
