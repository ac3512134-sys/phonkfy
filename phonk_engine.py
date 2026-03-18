#!/usr/bin/env python3
"""Simple phonk-style audio processor using pydub.

Pipeline:
1. Load segment.wav
2. Bass boost (low-frequency emphasis)
3. Add slight distortion (soft overdrive via clipping)
4. Slight speed increase
5. Normalize
6. Export phonk_output.wav
"""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path

INPUT_DEFAULT = Path("segment.wav")
OUTPUT_DEFAULT = Path("phonk_output.wav")


def load_audio(path: Path):
    """Load input audio file."""
    from pydub import AudioSegment

    return AudioSegment.from_file(str(path))


def bass_boost(audio, gain_db: float = 8.0, cutoff_hz: int = 150):
    """Boost low frequencies and blend with original signal."""
    low_band = audio.low_pass_filter(cutoff_hz).apply_gain(gain_db)
    return audio.overlay(low_band)


def apply_overdrive(audio, drive_db: float = 8.0):
    """Apply slight distortion by driving then hard-clipping samples."""
    driven = audio.apply_gain(drive_db)

    sample_width = driven.sample_width
    channels = driven.channels
    max_val = (1 << (8 * sample_width - 1)) - 1
    min_val = -(1 << (8 * sample_width - 1))

    samples = array(driven.array_type, driven.get_array_of_samples())

    # Gentle clipping threshold (~85% full scale) for slight distortion.
    clip_max = int(max_val * 0.85)
    clip_min = int(min_val * 0.85)

    for i, sample in enumerate(samples):
        if sample > clip_max:
            samples[i] = clip_max
        elif sample < clip_min:
            samples[i] = clip_min

    clipped = driven._spawn(samples.tobytes())
    return clipped.set_channels(channels)


def speed_up(audio, factor: float = 1.08):
    """Increase playback speed slightly for more aggression."""
    from pydub.effects import speedup

    return speedup(audio, playback_speed=factor, chunk_size=150, crossfade=25)


def normalize_audio(audio):
    """Normalize output level."""
    from pydub.effects import normalize

    return normalize(audio)


def process_phonk(
    input_path: Path = INPUT_DEFAULT,
    output_path: Path = OUTPUT_DEFAULT,
    speed_factor: float = 1.08,
) -> Path:
    """Run full phonk processing chain and export WAV output."""
    if not (1.05 <= speed_factor <= 1.1):
        raise ValueError("speed_factor must be between 1.05 and 1.1")

    audio = load_audio(input_path)
    boosted = bass_boost(audio)
    distorted = apply_overdrive(boosted)
    faster = speed_up(distorted, factor=speed_factor)
    final = normalize_audio(faster)

    final.export(str(output_path), format="wav")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply simple phonk-style processing.")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_DEFAULT,
        help='Input audio path (default: "segment.wav")',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DEFAULT,
        help='Output audio path (default: "phonk_output.wav")',
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.08,
        help="Playback speed factor between 1.05 and 1.1 (default: 1.08)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = process_phonk(input_path=args.input, output_path=args.output, speed_factor=args.speed)
    print(f'Phonk output saved to: "{out_path}"')


if __name__ == "__main__":
    main()
