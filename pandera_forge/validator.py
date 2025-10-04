"""
Model validation utilities
"""

from traceback import format_exc
from typing import Any, Dict, Optional
from pandas import DataFrame


class ModelValidator:
    """Validates generated Pandera models"""

    @staticmethod
    def validate_model_code(model_code: str, class_name: str) -> tuple[bool, Optional[str]]:
        """
        Validate that the generated model code is syntactically correct.

        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Create a namespace for execution
            namespace: Dict[str, Any] = {}
            # Add necessary imports to namespace
            exec(
                """
from pandera.pandas import Timestamp, DataFrameModel, Field
from pandera.typing import Series, Int64, Int32, Int16, Int8, Float64, Float32, Float16, String, Bool, DateTime, Category, Object
from typing import Optional
            """,
                namespace,
            )

            # Execute the model code
            exec(model_code, namespace)

            # Check if the class was created
            if class_name not in namespace:
                return False, f"Class {class_name} not found in generated code"

            return True, None

        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Execution error: {e}\n{format_exc()}"

    @staticmethod
    def validate_against_dataframe(
        model_code: str, class_name: str, df: DataFrame
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that the model can successfully validate the source DataFrame.

        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Create a namespace for execution
            namespace: Dict[str, Any] = {}

            # Add necessary imports
            exec(
                """
from pandera.pandas import Timestamp, DataFrameModel, Field
from pandera.typing.pandas import Series, Int64, Int32, Int16, Int8, Float64, Float32, Float16, String, Bool, DateTime, Category, Object
from typing import Optional
            """,
                namespace,
            )

            # Execute the model code
            exec(model_code, namespace)

            # Get the model class
            model_class = namespace.get(class_name)
            if model_class is None:
                return False, f"Class {class_name} not found"

            # Validate the DataFrame
            validated_df = model_class.validate(df)

            # Check if validation succeeded
            if validated_df is None:
                return False, "Validation returned None"

            return True, None

        except Exception as e:
            return False, f"Validation error: {e}\n{format_exc()}"
