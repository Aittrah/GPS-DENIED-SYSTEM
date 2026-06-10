import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np

logger = logging.getLogger("vns.validation")

@dataclass
class AccuracyThresholds:
    """Thresholds for validating VNS positioning accuracy."""
    max_mean_horizontal_error_m: float = 3.0
    max_p95_horizontal_error_m: float = 5.0
    max_mean_vertical_error_m: float = 2.0
    max_mean_heading_error_deg: float = 10.0

@dataclass
class AccuracyReport:
    """VNS accuracy evaluation report."""
    total_samples: int = 0
    duration_seconds: float = 0.0
    horizontal_error: Dict[str, float] = field(default_factory=dict)
    vertical_error: Dict[str, float] = field(default_factory=dict)
    heading_error: Dict[str, float] = field(default_factory=dict)
    passed: bool = True
    failures: List[str] = field(default_factory=list)

def load_and_generate_report(
    log_file: str,
    output_dir: str,
    thresholds: AccuracyThresholds
) -> AccuracyReport:
    """
    Load a JSONL log file, calculate positioning error stats, and generate markdown and json reports.
    """
    log_path = Path(log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")
        
    timestamps = []
    horiz_errors = []
    vert_errors = []
    heading_errors = []
    
    with open(log_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                timestamps.append(data.get("timestamp", 0.0))
                horiz_errors.append(data.get("horizontal_error", 0.0))
                vert_errors.append(data.get("vertical_error", 0.0))
                heading_errors.append(data.get("heading_error", 0.0))
            except Exception as e:
                logger.warning("Failed to parse log line: %s", e)
                
    if not horiz_errors:
        raise ValueError(f"No valid log samples found in: {log_file}")
        
    duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
    
    # Compute statistics
    horiz_mean = float(np.mean(horiz_errors))
    horiz_p95 = float(np.percentile(horiz_errors, 95))
    vert_mean = float(np.mean(vert_errors))
    heading_mean = float(np.mean(heading_errors))
    
    report = AccuracyReport(
        total_samples=len(horiz_errors),
        duration_seconds=duration,
        horizontal_error={
            "mean": horiz_mean,
            "p95": horiz_p95,
            "max": float(np.max(horiz_errors)),
            "min": float(np.min(horiz_errors))
        },
        vertical_error={
            "mean": vert_mean,
            "max": float(np.max(vert_errors)),
            "min": float(np.min(vert_errors))
        },
        heading_error={
            "mean": heading_mean,
            "max": float(np.max(heading_errors)),
            "min": float(np.min(heading_errors))
        }
    )
    
    # Validation against thresholds
    report.passed = True
    report.failures = []
    
    if horiz_mean > thresholds.max_mean_horizontal_error_m:
        report.passed = False
        report.failures.append(
            f"Mean horizontal error {horiz_mean:.3f}m exceeds threshold {thresholds.max_mean_horizontal_error_m:.3f}m"
        )
        
    if horiz_p95 > thresholds.max_p95_horizontal_error_m:
        report.passed = False
        report.failures.append(
            f"P95 horizontal error {horiz_p95:.3f}m exceeds threshold {thresholds.max_p95_horizontal_error_m:.3f}m"
        )
        
    if vert_mean > thresholds.max_mean_vertical_error_m:
        report.passed = False
        report.failures.append(
            f"Mean vertical error {vert_mean:.3f}m exceeds threshold {thresholds.max_mean_vertical_error_m:.3f}m"
        )
        
    if heading_mean > thresholds.max_mean_heading_error_deg:
        report.passed = False
        report.failures.append(
            f"Mean heading error {heading_mean:.3f}deg exceeds threshold {thresholds.max_mean_heading_error_deg:.3f}deg"
        )
        
    # Write reports
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    base_name = log_path.stem
    save_report(report, out_path, base_name)
    
    return report

def save_report(report: AccuracyReport, output_dir: Path, base_name: str) -> None:
    """Save the report as JSON and Markdown files."""
    # JSON output
    json_data = {
        "total_samples": report.total_samples,
        "duration_seconds": report.duration_seconds,
        "horizontal_error": report.horizontal_error,
        "vertical_error": report.vertical_error,
        "heading_error": report.heading_error,
        "passed": report.passed,
        "failures": report.failures
    }
    
    json_file = output_dir / f"report_{base_name}.json"
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=4)
        
    # Markdown output
    md_lines = [
        f"# VNS Accuracy Evaluation Report",
        f"",
        f"**Log File Source:** `{base_name}`  ",
        f"**Evaluation Status:** {'🟢 PASSED' if report.passed else '🔴 FAILED'}  ",
        f"**Total Samples:** `{report.total_samples}`  ",
        f"**Duration:** `{report.duration_seconds:.1f} seconds`  ",
        f"",
        f"## Positioning Accuracy Metrics",
        f"",
        f"| Metric | Mean | P95 | Max | Min |",
        f"|---|---|---|---|---|",
        f"| **Horizontal Error (m)** | {report.horizontal_error['mean']:.3f} | {report.horizontal_error['p95']:.3f} | {report.horizontal_error['max']:.3f} | {report.horizontal_error['min']:.3f} |",
        f"| **Vertical Error (m)** | {report.vertical_error['mean']:.3f} | N/A | {report.vertical_error['max']:.3f} | {report.vertical_error['min']:.3f} |",
        f"| **Heading Error (deg)** | {report.heading_error['mean']:.3f} | N/A | {report.heading_error['max']:.3f} | {report.heading_error['min']:.3f} |",
        f""
    ]
    
    if not report.passed:
        md_lines.extend([
            f"### Failure Causes",
            f""
        ])
        for failure in report.failures:
            md_lines.append(f"- 🔴 {failure}")
        md_lines.append("")
        
    md_file = output_dir / f"report_{base_name}.md"
    with open(md_file, "w") as f:
        f.write("\n".join(md_lines))
        
    logger.info("Saved accuracy reports to: %s and %s", json_file.name, md_file.name)
