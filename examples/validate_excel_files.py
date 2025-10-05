from pathlib import Path

from pandera_forge import ModelGenerator


# Generate the model
generator = ModelGenerator()

if __name__ == "__main__":
    downloads_folder = Path.home() / "Downloads"
    xlsx_files = list(downloads_folder.glob("*.xlsx"))
    # xls_files = list(downloads_folder.glob("*.xls"))
    # excel_files = xlsx_files + xls_files
    for excel_file in xlsx_files[1:5]:  # Limit to first 5 files for testing
        print(f"\n{'='*60}")
        print(f"Processing file: {excel_file.name}")
        print(f"{'='*60}")
        model_codes = generator.from_excel(excel_file, validate=True)
        for sheet_name, model_code in model_codes.items():
            print(f"\nSheet: {sheet_name}")
            if model_code:
                print(f"  ✓ Model generated and validated successfully!")
                print(f"\n{model_code[:500]}...")  # Show first 500 chars
            else:
                print("  ⚠️  Model generation or validation failed")
