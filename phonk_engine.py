#!/usr/bin/env python3
"""Aggressive phonk-style audio processor using pydub.

Pipeline:
1. Load segment.wav
2. Heavy bass boost using low-pass filtering + gain
3. Add distortion with gain and clipping
4. Add a simple dark reverb/echo tail
5. Speed up slightly
6. Normalize
7. Raise final loudness
8. Export phonk_output.wav
"""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path

INPUT_DEFAULT = Path("segment.wav")
OUTPUT_DEFAULT = Path("phonk_output.wav")
DEFAULT_SPEED = 1.08


def load_audio(path: Path):
    """Load input audio file with pydub."""
    from pydub import AudioSegment

    return AudioSegment.from_file(str(path))


def heavy_bass_boost(audio, gain_db: float = 12.0, cutoff_hz: int = 140):
    """Blend an aggressively boosted low-frequency layer back into the mix."""
    low_band = audio.low_pass_filter(cutoff_hz).apply_gain(gain_db)
    sub_band = audio.low_pass_filter(90).apply_gain(gain_db - 2.0)
    return audio.overlay(low_band).overlay(sub_band)


def apply_distortion(audio, drive_db: float = 14.0, clip_level: float = 0.72):
    """Add strong overdrive by driving the signal and clipping the peaks."""
    driven = audio.apply_gain(drive_db)

    sample_width = driven.sample_width
    samples = array(driven.array_type, driven.get_array_of_samples())

    max_val = (1 << (8 * sample_width - 1)) - 1
    min_val = -(1 << (8 * sample_width - 1))
    clip_max = int(max_val * clip_level)
    clip_min = int(min_val * clip_level)

    for index, sample in enumerate(samples):
        if sample > clip_max:
            samples[index] = clip_max
        elif sample < clip_min:
            samples[index] = clip_min

    return driven._spawn(samples.tobytes())


def add_reverb(audio, delay_ms: int = 120, decay_db: float = 7.0, repeats: int = 3):
    """Create a simple echo-style reverb by layering delayed, quieter copies."""
    wet = audio

    for repeat in range(1, repeats + 1):
        delayed = audio.apply_gain(-(decay_db * repeat))
        wet = wet.overlay(delayed, position=delay_ms * repeat)

    return wet


def speed_up(audio, factor: float = DEFAULT_SPEED):
    """Increase playback speed for extra urgency and bounce."""
    from pydub.effects import speedup

    return speedup(audio, playback_speed=factor, chunk_size=120, crossfade=20)


def normalize_audio(audio):
    """Normalize the processed signal."""
    from pydub.effects import normalize

    return normalize(audio)


def make_louder(audio, gain_db: float = 3.5):
    """Raise final loudness after normalization."""
    return audio.apply_gain(gain_db)


def process_phonk(
    input_path: Path = INPUT_DEFAULT,
    output_path: Path = OUTPUT_DEFAULT,
    speed_factor: float = DEFAULT_SPEED,
) -> Path:
    """Run the full aggressive phonk processing chain and export output."""
    if abs(speed_factor - 1.08) > 0.02:
        raise ValueError("speed_factor must stay close to 1.08 (between 1.06 and 1.10)")

    audio = load_audio(input_path)
    bassy = heavy_bass_boost(audio)
    distorted = apply_distortion(bassy)
    reverbed = add_reverb(distorted)
    faster = speed_up(reverbed, factor=speed_factor)
    normalized = normalize_audio(faster)
    final = make_louder(normalized)

    final.export(str(output_path), format="wav")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Apply aggressive phonk-style processing.")
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
        default=DEFAULT_SPEED,
        help="Playback speed factor, intended around 1.08 (default: 1.08)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    out_path = process_phonk(input_path=args.input, output_path=args.output, speed_factor=args.speed)
    print(f'Phonk output saved to: "{out_path}"')


if __name__ == "__main__":
    main()
