"""
Script to inspect the schema definition at runtime.
"""
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import the schema module
try:
    from app.models import schemas
    
    # Print the schema definition
    print("PredictionInput schema:")
    print("-" * 50)
    print(schemas.PredictionInput.schema_json(indent=2))
    
    # Print required fields
    print("\nRequired fields:")
    print("-" * 50)
    print(schemas.PredictionInput.schema().get("required", []))
    
except Exception as e:
    print(f"Error importing schema: {e}")
    
    # Try to print the schema file directly
    try:
        with open("app/models/schemas.py", "r") as f:
            print("\nSchema file content:")
            print("-" * 50)
            print(f.read())
    except Exception as e2:
        print(f"Error reading schema file: {e2}")
