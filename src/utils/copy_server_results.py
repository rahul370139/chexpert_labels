#!/usr/bin/env python3
"""
Copy and convert server CheXagent results to local.

Converts server paths to local paths:
  /Users/bilbouser/radiology_report/... → /Users/rahul/Downloads/Code scripts/radiology_report/...
"""

import argparse
import pandas as pd
import subprocess
import sys
from pathlib import Path


def copy_and_convert_server_csv(
    server_host: str,
    server_path: str,
    local_output: Path,
    local_radiology_root: Path,
) -> None:
    """
    Copy CSV from server and convert paths.
    
    Args:
        server_host: SSH host (e.g., "bilbouser@100.77.217.18")
        server_path: Remote path (e.g., "~/chexagent_chexpert_eval/chex_full/hybrid_full.csv")
        local_output: Local output path
        local_radiology_root: Local radiology_report root directory
    """
    print("=" * 80)
    print("COPYING SERVER CHEXAGENT RESULTS")
    print("=" * 80)
    
    # Step 1: Copy from server
    print(f"\n📥 Copying from server...")
    print(f"   Server: {server_host}:{server_path}")
    
    temp_csv = local_output.parent / f".temp_{local_output.name}"
    
    # Use scp to copy
    scp_cmd = [
        "scp",
        f"{server_host}:{server_path}",
        str(temp_csv)
    ]
    
    result = subprocess.run(scp_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to copy from server:")
        print(f"   {result.stderr}")
        sys.exit(1)
    
    print(f"   ✅ Copied to temporary file: {temp_csv}")
    
    # Step 2: Load and validate
    print(f"\n🔍 Loading and validating CSV...")
    df = pd.read_csv(temp_csv)
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)[:10]}...")  # Show first 10
    
    # Assert required columns
    required_cols = ["image", "binary_outputs", "di_outputs"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"❌ Missing required columns: {missing}")
        print(f"   Expected: {required_cols}")
        print(f"   Found: {list(df.columns)}")
        sys.exit(1)
    
    print(f"   ✅ Required columns present: {required_cols}")
    
    # Step 3: Convert paths
    print(f"\n🔄 Converting server paths to local paths...")
    server_prefix = "/Users/bilbouser/radiology_report"
    local_prefix = str(local_radiology_root.absolute())
    
    converted = 0
    for idx, path in enumerate(df["image"]):
        if pd.notna(path) and str(path).startswith(server_prefix):
            local_path = str(path).replace(server_prefix, local_prefix)
            df.at[idx, "image"] = local_path
            converted += 1
    
    print(f"   Converted {converted} paths")
    
    # Step 4: Add filename column (standardized join key)
    df["filename"] = df["image"].apply(lambda x: Path(x).name if pd.notna(x) else "")
    
    # Step 5: Save
    local_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(local_output, index=False)
    print(f"\n✅ Saved converted CSV to: {local_output}")
    print(f"   Rows: {len(df)}")
    
    # Cleanup temp file
    temp_csv.unlink()
    
    print("\n" + "=" * 80)
    print("COPY COMPLETE")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Copy and convert server CheXagent results")
    parser.add_argument(
        "--server_host",
        default="bilbouser@100.77.217.18",
        help="SSH host for server"
    )
    parser.add_argument(
        "--server_path",
        default="~/chexagent_chexpert_eval/chex_full/hybrid_full.csv",
        help="Remote CSV path"
    )
    parser.add_argument(
        "--local_output",
        type=Path,
        default=Path("results/hybrid_ensemble_5826.csv"),
        help="Local output CSV path"
    )
    parser.add_argument(
        "--radiology_root",
        type=Path,
        default=Path("../radiology_report"),
        help="Local radiology_report root directory"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent.parent
    local_output = (project_root / args.local_output).resolve()
    radiology_root = (project_root / args.radiology_root).resolve()
    
    if not radiology_root.exists():
        print(f"❌ Radiology root not found: {radiology_root}")
        sys.exit(1)
    
    copy_and_convert_server_csv(
        args.server_host,
        args.server_path,
        local_output,
        radiology_root,
    )


if __name__ == "__main__":
    main()

