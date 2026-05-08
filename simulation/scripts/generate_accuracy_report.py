#!/usr/bin/env python3
"""
Generate Accuracy Report from VNS Validation Logs

This script processes ground truth log files and generates comprehensive
accuracy reports for VNS validation.

Usage:
    python generate_accuracy_report.py --input logs/ground_truth_*.jsonl --output reports/
    
    # With custom thresholds:
    python generate_accuracy_report.py --input log.jsonl --max-horizontal-error 2.0
"""

import argparse
import glob
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from vns.validation.accuracy_report import (
    AccuracyThresholds,
    load_and_generate_report,
    save_report,
)


def main():
    parser = argparse.ArgumentParser(
        description='Generate VNS accuracy reports from validation logs'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to log file(s), supports glob patterns'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./reports',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--max-horizontal-error',
        type=float,
        default=3.0,
        help='Maximum acceptable mean horizontal error (meters)'
    )
    parser.add_argument(
        '--max-p95-error',
        type=float,
        default=5.0,
        help='Maximum acceptable P95 horizontal error (meters)'
    )
    parser.add_argument(
        '--max-vertical-error',
        type=float,
        default=2.0,
        help='Maximum acceptable mean vertical error (meters)'
    )
    parser.add_argument(
        '--max-heading-error',
        type=float,
        default=10.0,
        help='Maximum acceptable mean heading error (degrees)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'markdown', 'both'],
        default='both',
        help='Output format'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Print summary to console'
    )
    
    args = parser.parse_args()
    
    # Find log files
    log_files = glob.glob(args.input)
    if not log_files:
        print(f"Error: No files found matching '{args.input}'")
        sys.exit(1)
    
    print(f"Found {len(log_files)} log file(s)")
    
    # Create thresholds
    thresholds = AccuracyThresholds(
        max_mean_horizontal_error_m=args.max_horizontal_error,
        max_p95_horizontal_error_m=args.max_p95_error,
        max_mean_vertical_error_m=args.max_vertical_error,
        max_mean_heading_error_deg=args.max_heading_error
    )
    
    # Process each log file
    all_passed = True
    
    for log_file in log_files:
        print(f"\nProcessing: {log_file}")
        
        try:
            report = load_and_generate_report(
                log_file=log_file,
                output_dir=args.output,
                thresholds=thresholds
            )
            
            if args.summary:
                print(f"  Samples: {report.total_samples}")
                print(f"  Duration: {report.duration_seconds:.1f}s")
                print(f"  Mean Horizontal Error: {report.horizontal_error.get('mean', 0):.3f}m")
                print(f"  P95 Horizontal Error: {report.horizontal_error.get('p95', 0):.3f}m")
                print(f"  Result: {'PASSED' if report.passed else 'FAILED'}")
                
                if report.failures:
                    for failure in report.failures:
                        print(f"    - {failure}")
            
            if not report.passed:
                all_passed = False
                
        except Exception as e:
            print(f"  Error: {e}")
            all_passed = False
    
    print(f"\nReports saved to: {args.output}")
    print(f"Overall Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()