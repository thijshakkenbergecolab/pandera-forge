from pathlib import Path

from pandera_forge import ModelGenerator


# Generate the model
generator = ModelGenerator()

if __name__ == "__main__":
    downloads_folder = Path.home() / "Downloads"
    csv_files = list(downloads_folder.glob("*.csv"))

    # Limit to first 5 files for testing
    for csv_file in csv_files[:5]:
        print(f"\n{'='*60}")
        print(f"Processing file: {csv_file.name}")
        print(f"{'='*60}")
        model_code = generator.from_csv(csv_file, validate=True)
        if model_code:
            print(f"\n  ✓ Model generated and validated successfully!")
            print(f"\n{model_code[:500]}...")  # Show first 500 chars
