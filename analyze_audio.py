#!/usr/bin/env python3
"""Extract the loudest continuous audio segment using pydub.

Workflow:
1. Load MP3/WAV audio
2. Convert to mono
3. Split into fixed-size chunks (default 500 ms)
4. Measure chunk loudness (dBFS)
5. Find the loudest continuous 10-15 second segment
6. Export that segment as WAV
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OUTPUT = Path("segment.wav")
DEFAULT_CHUNK_MS = 500
DEFAULT_SEGMENT_SECONDS = 12  # must remain within 10-15
MIN_SEGMENT_SECONDS = 10
MAX_SEGMENT_SECONDS = 15


def load_audio_mono(input_path: Path):
    """Load audio file with pydub and convert it to mono."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(input_path))
    return audio.set_channels(1)


def chunk_loudness_dbfs(audio, chunk_ms: int) -> list[float]:
    """Compute dBFS loudness for each chunk."""
    loudness_values: list[float] = []

    for start in range(0, len(audio), chunk_ms):
        chunk = audio[start : start + chunk_ms]

        # Silence can report -inf dBFS; keep it very low but finite for math.
        value = chunk.dBFS if chunk.dBFS != float("-inf") else -120.0
        loudness_values.append(value)

    return loudness_values


def find_loudest_segment_bounds(
    loudness_values: list[float],
    total_duration_ms: int,
    chunk_ms: int,
    segment_seconds: int,
) -> tuple[int, int]:
    """Return [start_ms, end_ms] of the loudest continuous segment."""
    segment_ms = segment_seconds * 1000

    # If audio is shorter than desired segment, return full audio.
    if total_duration_ms <= segment_ms:
        return 0, total_duration_ms

    window_chunks = max(1, segment_ms // chunk_ms)

    # Sliding-window sum over chunk loudness values.
    current_sum = sum(loudness_values[:window_chunks])
    best_sum = current_sum
    best_start_chunk = 0

    for i in range(window_chunks, len(loudness_values)):
        current_sum += loudness_values[i] - loudness_values[i - window_chunks]
        if current_sum > best_sum:
            best_sum = current_sum
            best_start_chunk = i - window_chunks + 1

    start_ms = best_start_chunk * chunk_ms
    end_ms = start_ms + segment_ms

    # Clip to audio duration.
    if end_ms > total_duration_ms:
        end_ms = total_duration_ms
        start_ms = max(0, end_ms - segment_ms)

    return start_ms, end_ms


def extract_loudest_segment(
    input_path: Path,
    output_path: Path = DEFAULT_OUTPUT,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    segment_seconds: int = DEFAULT_SEGMENT_SECONDS,
) -> Path:
    """Run full extraction pipeline and export loudest segment as WAV."""
    if segment_seconds < MIN_SEGMENT_SECONDS or segment_seconds > MAX_SEGMENT_SECONDS:
        raise ValueError("segment_seconds must be between 10 and 15.")
    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be positive.")

    audio = load_audio_mono(input_path)
    loudness_values = chunk_loudness_dbfs(audio, chunk_ms)

    start_ms, end_ms = find_loudest_segment_bounds(
        loudness_values=loudness_values,
        total_duration_ms=len(audio),
        chunk_ms=chunk_ms,
        segment_seconds=segment_seconds,
    )

    segment = audio[start_ms:end_ms]
    segment.export(str(output_path), format="wav")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Extract the loudest continuous 10-15s segment from mp3/wav."
    )
    parser.add_argument("input_audio", type=Path, help="Input audio file path (mp3 or wav).")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help='Output path (default: "segment.wav").',
    )
    parser.add_argument(
        "--segment-seconds",
        type=int,
        default=DEFAULT_SEGMENT_SECONDS,
        help="Target segment length in seconds (10-15).",
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=DEFAULT_CHUNK_MS,
        help="Chunk size in milliseconds for loudness analysis (default: 500).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = extract_loudest_segment(
        input_path=args.input_audio,
        output_path=args.output,
        chunk_ms=args.chunk_ms,
        segment_seconds=args.segment_seconds,
    )
    print(f'Extracted segment saved to: "{output}"')


if __name__ == "__main__":
    main()
