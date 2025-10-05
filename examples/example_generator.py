"""
Test script to verify the refactored generator works correctly
"""

from pandas import DataFrame, to_datetime, Categorical
from pandera_forge import ModelGenerator


def test_basic_generation():
    """Test basic model generation with various data types"""
    # Create test DataFrame with various data types
    df = DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [1.0, 2.0, 3.0, 4.0],
            "C": ["a", "b", None, "d"],
            "D": [True, False, True, False],
            "E": to_datetime(["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"]),
            3: [1, 2, 3, 4],  # Numeric column name
            "F": Categorical(["a", "bb", "ccc", "dddd"]),
            "G": ["a", "b", "c", "d"],
            "I have spaces": ["a", "b", "c", "d"],
            "J": ["a", "word", "this is a sentence", "this is a sentence with a number 1"],
            "class": [1, 2, 3, 4],  # Reserved keyword
            "!@#$%": [5, 6, 7, 8],  # Special characters
        }
    )

    # Generate model
    generator = ModelGenerator()
    model_code = generator.generate(df, "TestModel", validate=True)

    if model_code:
        print("✅ Model generated successfully!")
        print("\n" + "=" * 60)
        print("Generated Model:")
        print("=" * 60)
        print(model_code)
        print("=" * 60 + "\n")

        # Test that the generated code can be executed
        try:
            exec(model_code)
            print("✅ Generated code executes without errors")
        except Exception as e:
            print(f"❌ Error executing generated code: {e}")
    else:
        print("❌ Model generation failed")

    return model_code


def test_edge_cases():
    """Test edge cases and problematic data"""
    # DataFrame with edge cases
    df_edge = DataFrame(
        {
            "empty_col": [None, None, None],
            "mixed_types": [1, "two", 3.0],  # This will be object type
            "all_unique": [1, 2, 3],
            "all_same": [1, 1, 1],
            123: ["numeric", "column", "name"],
            "": ["empty", "column", "name"],  # Empty string column name
        }
    )

    generator = ModelGenerator()
    model_code = generator.generate(df_edge, "EdgeCaseModel", validate=True)

    if model_code:
        print("✅ Edge case model generated successfully!")
        print("\n" + "=" * 60)
        print("Edge Case Model:")
        print("=" * 60)
        print(model_code)
        print("=" * 60 + "\n")
    else:
        print("⚠️  Edge case model generation failed (this might be expected)")

    return model_code


if __name__ == "__main__":
    print("Testing Pandere Forge Generator")
    print("=" * 60)

    # Run basic test
    basic_model = test_basic_generation()

    print("\n")

    # Run edge case test
    edge_model = test_edge_cases()
    if edge_model:
        print("Edge Model Code:")
        print(edge_model)
    print("\n✅ All tests completed!")
