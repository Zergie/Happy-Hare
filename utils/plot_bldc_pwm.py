from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

DEFAULT_INPUT = Path(".github/skills/run-test-rig/klippy.log")
DEFAULT_UNIT = "mmu_gear_bldc unit0"
SECTION_HEADER_PATTERN = "[{unit}]"

BLDC_PIN_PATTERN = re.compile(
    r"(?m)^.*BLDC_SET_PIN: (?P<message>.+?) applied=(?P<applied>[0-9.]+)(?: time=(?P<time>[0-9.]+))?.* unit=(?P<unit>.+)$"
)
BLDC_TACH_PATTERN = re.compile(
    r"^.*BLDC_TACH: freq=(?P<freq>[0-9.]+) rpm=(?P<rpm>[0-9.]+)(?: time=(?P<time>[0-9.]+))?.* unit=(?P<unit>.+)$"
)
ROTATION_DISTANCE_PATTERN = re.compile(r"^rotation_distance\s*=\s*(?P<rotation_distance>[0-9.]+)\s*$")


@dataclass(frozen=True)
class BldcPwmSample:
    time_seconds: float
    pwm: float


@dataclass(frozen=True)
class BldcRpmSample:
    time_seconds: float
    rpm: float


def parse_bldc_pwm_samples(log_text: str, unit: str) -> list[BldcPwmSample]:
    samples: list[BldcPwmSample] = []
    for match in BLDC_PIN_PATTERN.finditer(log_text):
        matched_unit = match.group("unit").strip()
        if matched_unit != unit:
            continue

        message = match.group("message").strip()
        if "PWM" not in message:
            continue

        time_value = match.group("time")
        if not time_value:
            continue

        samples.append(
            BldcPwmSample(
                time_seconds=float(time_value),
                pwm=float(match.group("applied")),
            )
        )

    samples.sort(key=lambda sample: sample.time_seconds)
    return samples


def parse_bldc_rpm_samples(log_text: str, unit: str) -> list[BldcRpmSample]:
    samples: list[BldcRpmSample] = []
    last_time_for_unit: float | None = None

    for raw_line in log_text.splitlines():
        pin_match = BLDC_PIN_PATTERN.match(raw_line)
        if pin_match:
            pin_unit = pin_match.group("unit").strip()
            pin_time = pin_match.group("time")
            if pin_unit == unit and pin_time:
                last_time_for_unit = float(pin_time)

        tach_match = BLDC_TACH_PATTERN.match(raw_line)
        if not tach_match:
            continue

        tach_unit = tach_match.group("unit").strip()
        if tach_unit != unit:
            continue

        tach_time = tach_match.group("time")
        sample_time = None
        if tach_time:
            sample_time = float(tach_time)
        elif last_time_for_unit is not None:
            sample_time = last_time_for_unit
        if sample_time is None:
            continue

        samples.append(
            BldcRpmSample(
                time_seconds=sample_time,
                rpm=float(tach_match.group("rpm")),
            )
        )

    samples.sort(key=lambda sample: sample.time_seconds)
    return samples


def get_bldc_rotation_distance(log_text: str, unit: str) -> float:
    section_header = SECTION_HEADER_PATTERN.format(unit=unit)
    in_target_section = False

    for raw_line in log_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_target_section = line == section_header
            continue
        if not in_target_section:
            continue

        match = ROTATION_DISTANCE_PATTERN.match(line)
        if match:
            return float(match.group("rotation_distance"))

    raise RuntimeError(f"Missing rotation_distance for unit '{unit}' in log")


def to_relative_time(samples: list[BldcPwmSample]) -> list[BldcPwmSample]:
    if not samples:
        return []

    t0 = samples[0].time_seconds
    return [BldcPwmSample(time_seconds=sample.time_seconds - t0, pwm=sample.pwm) for sample in samples]


def to_relative_rpm_time(samples: list[BldcRpmSample], t0: float) -> list[BldcRpmSample]:
    return [BldcRpmSample(time_seconds=sample.time_seconds - t0, rpm=sample.rpm) for sample in samples]


def rpm_to_speed_mm_s(rpm: float, rotation_distance: float) -> float:
    return rpm * rotation_distance / 60.


def plot_bldc_metrics(
    pwm_samples: list[BldcPwmSample],
    rpm_samples: list[BldcRpmSample],
    unit: str,
    use_absolute_time: bool,
    rotation_distance: float,
) -> None:
    title_suffix = "absolute" if use_absolute_time else "relative"

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    pwm_x = [sample.time_seconds for sample in pwm_samples]
    pwm_y = [sample.pwm for sample in pwm_samples]
    axes[0].plot(pwm_x, pwm_y, linewidth=1.25, label=f"{unit} PWM")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("PWM")
    axes[0].set_title(f"BLDC PWM and reported speed over time ({title_suffix} time)")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="best")

    if rpm_samples:
        rpm_x = [sample.time_seconds for sample in rpm_samples]
        speed_y = [rpm_to_speed_mm_s(sample.rpm, rotation_distance) for sample in rpm_samples]
        axes[1].plot(rpm_x, speed_y, linewidth=1.25, color="tab:orange", label=f"{unit} reported speed")
        axes[1].legend(loc="best")
    else:
        axes[1].text(0.5, 0.5, "No BLDC_TACH speed samples found", ha="center", va="center", transform=axes[1].transAxes)

    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed (mm/s)")
    axes[1].grid(True, alpha=0.35)

    fig.tight_layout()

    plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BLDC PWM over time from klippy.log")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to klippy log file")
    parser.add_argument("--unit", default=DEFAULT_UNIT, help="BLDC unit section name")
    parser.add_argument(
        "--rotation-distance",
        type=float,
        help="Rotation distance in mm per revolution; defaults to value parsed from selected BLDC section in log",
    )
    parser.add_argument(
        "--absolute-time",
        action="store_true",
        help="Use absolute print time from log instead of relative time from first sample",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input log not found: {args.input}")

    log_text = args.input.read_text(encoding="utf-8")
    samples = parse_bldc_pwm_samples(log_text, unit=args.unit)
    if not samples:
        raise RuntimeError(f"No BLDC PWM samples found for unit '{args.unit}' in {args.input}")

    rpm_samples = parse_bldc_rpm_samples(log_text, unit=args.unit)
    rotation_distance = args.rotation_distance or get_bldc_rotation_distance(log_text, unit=args.unit)

    plot_samples = samples if args.absolute_time else to_relative_time(samples)
    plot_rpm_samples = rpm_samples
    if not args.absolute_time and rpm_samples:
        plot_rpm_samples = to_relative_rpm_time(rpm_samples, t0=samples[0].time_seconds)

    plot_bldc_metrics(
        plot_samples,
        plot_rpm_samples,
        unit=args.unit,
        use_absolute_time=args.absolute_time,
        rotation_distance=rotation_distance,
    )

    first_sample = plot_samples[0]
    last_sample = plot_samples[-1]
    print(
        f"Plotted {len(plot_samples)} PWM samples and {len(plot_rpm_samples)} speed samples. "
        f"first=({first_sample.time_seconds:.3f}s,{first_sample.pwm:.4f}) "
        f"last=({last_sample.time_seconds:.3f}s,{last_sample.pwm:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
