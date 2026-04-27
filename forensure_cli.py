import argparse
import sys
import os
from pathlib import Path

from src.triage.inference import ForensicScanner
from src.evaluation.report_generator import generate_html_report

def main():
    parser = argparse.ArgumentParser(description="ForenSURE-Net Forensic Triage Tool")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan a directory of images and generate a report")
    scan_parser.add_argument("--dir", type=str, required=True, help="Path to the directory containing images to scan")
    scan_parser.add_argument("--weights", type=str, required=True, help="Path to the trained model weights (.pth)")
    scan_parser.add_argument("--temperature", type=str, default=None, help="Path to the calibration temperature weights (optional)")
    scan_parser.add_argument("--out", type=str, default="forensic_report.html", help="Path to save the output HTML report")
    
    args = parser.parse_args()
    
    if args.command == "scan":
        if not os.path.exists(args.dir):
            print(f"Error: Input directory '{args.dir}' does not exist.")
            sys.exit(1)
            
        if not os.path.exists(args.weights):
            print(f"Error: Model weights file '{args.weights}' does not exist.")
            sys.exit(1)
            
        print(f"=========================================")
        print(f" ForenSURE-Net Forensic Scanner ")
        print(f"=========================================")
        print(f"Target Dir: {args.dir}")
        print(f"Model Wgts: {args.weights}")
        print(f"Output Rpt: {args.out}")
        print(f"=========================================\\n")
        
        # Initialize Scanner
        scanner = ForensicScanner(
            weights_path=args.weights,
            temperature_path=args.temperature
        )
        
        # Run Scan
        results = scanner.scan_directory(args.dir)
        
        if not results:
            print("Scan completed but no results to report.")
            sys.exit(0)
            
        # Generate Report
        print(f"\\nGenerating report with {len(results)} items...")
        generate_html_report(results, args.out, str(Path(args.dir).absolute()))
        
        print(f"\\n[SUCCESS] Forensic report saved to: {args.out}")
        print("Open this file in any web browser to view the triage results.")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
