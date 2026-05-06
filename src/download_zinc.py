#!/usr/bin/env python3
"""
Download ZINC 250k dataset for pre-training chemical VAE.

This script fetches the canonical ZINC 250k curated dataset from the
chemical_vae repository and saves it locally for graph preprocessing.

URL: https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv
Output: data/raw/zinc250k.csv
"""

import os
import sys
import requests
from pathlib import Path
from tqdm import tqdm


def download_zinc_250k(output_dir: str = "data/raw", filename: str = "zinc250k.csv"):
    """
    Download ZINC 250k dataset with progress bar.
    
    Args:
        output_dir: Directory to save the dataset (default: data/raw)
        filename: Output filename (default: zinc250k.csv)
    
    Returns:
        Path to downloaded file if successful, None otherwise
    """
    # Dataset URL
    url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
    
    # Create output directory if needed
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Full output file path
    file_path = output_path / filename
    
    print(f"📥 Downloading ZINC 250k dataset...")
    print(f"   URL: {url}")
    print(f"   Destination: {file_path}")
    print()
    
    try:
        # Stream the response to show progress
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size == 0:
            print("⚠️  Warning: Content-length header not provided, progress bar will be indeterminate")
        
        # Download with progress bar
        with open(file_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify file was created
        if file_path.exists():
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print()
            print(f"✅ Successfully downloaded ZINC 250k dataset!")
            print(f"   File: {file_path}")
            print(f"   Size: {file_size_mb:.2f} MB")
            return str(file_path)
        else:
            print(f"❌ Error: File was not created at {file_path}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"❌ Error: Download timed out. Please try again.")
        return None
    except requests.exceptions.ConnectionError:
        print(f"❌ Error: Connection failed. Please check your internet connection.")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"❌ Error: HTTP error {e.response.status_code}")
        if e.response.status_code == 404:
            print("   The URL may no longer be available.")
        return None
    except Exception as e:
        print(f"❌ Error: Failed to download dataset: {e}")
        return None


if __name__ == "__main__":
    output_dir = "data/raw"
    filename = "zinc250k.csv"
    
    # Allow command-line overrides
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    if len(sys.argv) > 2:
        filename = sys.argv[2]
    
    file_path = download_zinc_250k(output_dir=output_dir, filename=filename)
    
    if file_path:
        print()
        print("🎉 ZINC 250k dataset ready for preprocessing!")
        sys.exit(0)
    else:
        sys.exit(1)
