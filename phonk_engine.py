#!/usr/bin/env python3
"""Phonk-style enhancement processor using pydub.

This version keeps the original song intact and enhances it with:
1. Bass reinforcement
2. Mild overdrive
3. Subtle echo/reverb
4. Slight speed increase
5. Normalization
"""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path

INPUT_DEFAULT = Path("segment.wav")
OUTPUT_DEFAULT = Path("phonk_output.wav")
DEFAULT_SPEED = 1.05


def load_audio(path: Path):
    """Load input audio from disk."""
    from pydub import AudioSegment

    return AudioSegment.from_file(str(path))


def create_bass_layer(audio, cutoff_hz: int = 150, bass_gain_db: float = 6.0):
    """Create a reinforced low-frequency layer from the original audio."""
    return audio.low_pass_filter(cutoff_hz).apply_gain(bass_gain_db)


def mix_bass_with_original(audio, bass_layer, bass_blend_db: float = -2.0):
    """Blend the enhanced bass layer back into the original mix."""
    return audio.overlay(bass_layer.apply_gain(bass_blend_db))


def apply_mild_overdrive(audio, drive_db: float = 4.0, clip_level: float = 0.9):
    """Apply gentle clipping to add grit without destroying the melody."""
    driven = audio.apply_gain(drive_db)
    samples = array(driven.array_type, driven.get_array_of_samples())
    sample_width = driven.sample_width

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


def add_subtle_reverb(audio, delay_ms: int = 90, decay_db: float = 10.0):
    """Add a light echo to give space without muddying the track."""
    echo = audio.apply_gain(-decay_db)
    return audio.overlay(echo, position=delay_ms)


def speed_up(audio, factor: float = DEFAULT_SPEED):
    """Apply a small speed increase while preserving musical clarity."""
    from pydub.effects import speedup

    return speedup(audio, playback_speed=factor, chunk_size=150, crossfade=25)


def normalize_audio(audio):
    """Normalize the final result."""
    from pydub.effects import normalize

    return normalize(audio)


def process_phonk(
    input_path: Path = INPUT_DEFAULT,
    output_path: Path = OUTPUT_DEFAULT,
    speed_factor: float = DEFAULT_SPEED,
) -> Path:
    """Enhance the original segment with phonk-style processing."""
    if abs(speed_factor - 1.05) > 0.02:
        raise ValueError("speed_factor must stay near 1.05 (between 1.03 and 1.07)")

    original = load_audio(input_path)
    bass_layer = create_bass_layer(original)
    enhanced = mix_bass_with_original(original, bass_layer)
    driven = apply_mild_overdrive(enhanced)
    reverbed = add_subtle_reverb(driven)
    faster = speed_up(reverbed, factor=speed_factor)
    final = normalize_audio(faster)

    final.export(str(output_path), format="wav")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Enhance segment.wav with a clear phonk-style mix.")
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
        help="Playback speed factor, intended around 1.05 (default: 1.05)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    output_path = process_phonk(
        input_path=args.input,
        output_path=args.output,
        speed_factor=args.speed,
    )
    print(f'Phonk output saved to: "{output_path}"')


if __name__ == "__main__":
    main()
