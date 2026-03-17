#!/usr/bin/env python3
"""Audio analysis utility for phonk preprocessing.

This script loads an input audio file, estimates tempo (BPM), computes loudness
(RMS energy) over time, finds the most energetic region, and extracts a
10-15 second segment around that region.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

# Analysis parameters
FRAME_LENGTH = 2048
HOP_LENGTH = 512
DEFAULT_SEGMENT_SECONDS = 12.0  # must stay between 10 and 15 seconds


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file preserving its native sample rate.

    Args:
        audio_path: Path to an audio file (e.g. WAV or MP3).

    Returns:
        A tuple of (audio_samples_mono, sample_rate).
    """
    import librosa

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    return y, sr


def detect_bpm(y: np.ndarray, sr: int) -> float:
    """Detect track tempo in BPM using librosa beat tracking."""
    import librosa
    import numpy as np

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(np.atleast_1d(tempo)[0])


def compute_rms_energy(y: np.ndarray) -> np.ndarray:
    """Compute RMS energy per analysis frame."""
    import librosa

    return librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]


def get_segment_bounds(
    y: np.ndarray,
    sr: int,
    rms: np.ndarray,
    segment_seconds: float,
) -> tuple[int, int]:
    """Get sample-accurate segment bounds around the most energetic region.

    The segment is centered around the highest-energy frame and clipped to the
    valid audio range.
    """
    total_samples = len(y)
    segment_samples = int(segment_seconds * sr)

    # If audio is shorter than target segment duration, return full audio.
    if segment_samples >= total_samples:
        return 0, total_samples

    import numpy as np

    peak_frame = int(np.argmax(rms))
    peak_sample = peak_frame * HOP_LENGTH

    half = segment_samples // 2
    start = peak_sample - half
    end = start + segment_samples

    # Clip bounds to valid audio duration.
    if start < 0:
        start = 0
        end = segment_samples
    if end > total_samples:
        end = total_samples
        start = total_samples - segment_samples

    return int(start), int(end)


def extract_segment(y: np.ndarray, start: int, end: int) -> np.ndarray:
    """Extract audio segment from sample indices."""
    return y[start:end]


def analyze_and_extract(
    audio_path: Path,
    output_path: Path = Path("segment.wav"),
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
) -> float:
    """Run complete analysis pipeline and save extracted segment.

    Returns:
        Detected BPM.
    """
    if not (10.0 <= segment_seconds <= 15.0):
        raise ValueError("segment_seconds must be between 10 and 15 seconds.")

    y, sr = load_audio(audio_path)
    bpm = detect_bpm(y, sr)
    rms = compute_rms_energy(y)

    start, end = get_segment_bounds(y, sr, rms, segment_seconds)
    segment = extract_segment(y, start, end)

    import soundfile as sf

    sf.write(str(output_path), segment, sr)

    return bpm


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze an audio file and export the most energetic segment."
    )
    parser.add_argument(
        "input_audio",
        type=Path,
        help="Input audio file path (wav or mp3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("segment.wav"),
        help='Output path for extracted segment (default: "segment.wav").',
    )
    parser.add_argument(
        "--segment-seconds",
        type=float,
        default=DEFAULT_SEGMENT_SECONDS,
        help="Segment duration in seconds (must be between 10 and 15).",
    )
    return parser.parse_args()


def main() -> None:
    """Script entry point."""
    args = parse_args()
    bpm = analyze_and_extract(
        audio_path=args.input_audio,
        output_path=args.output,
        segment_seconds=args.segment_seconds,
    )

    print(f"Detected BPM: {bpm:.2f}")
    print(f'Extracted segment saved to: "{args.output}"')


if __name__ == "__main__":
    main()
