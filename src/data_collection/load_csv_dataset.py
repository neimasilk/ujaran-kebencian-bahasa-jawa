import pandas as pd
import os

# Define the relative path to the dataset
DATASET_PATH = "data/raw/raw-dataset.csv"
REPORT_PATH = "memory-bank/dataset_inspection_report.txt"

def inspect_dataset(dataset_path):
    """
    Loads a CSV dataset using Pandas, performs basic inspection,
    and returns the inspection details as a string.

    Args:
        dataset_path (str): The path to the CSV dataset.

    Returns:
        str: A string containing the dataset inspection report,
             or an error message if the file is not found or is not a CSV.
    """
    report_lines = []
    report_lines.append(f"Inspecting dataset: {dataset_path}\n")

    if not os.path.exists(dataset_path):
        return f"Error: Dataset file not found at {dataset_path}"

    try:
        # Load the dataset
        df = pd.read_csv(dataset_path)
        report_lines.append("Dataset loaded successfully.\n")

        # Basic Information
        report_lines.append("--- Basic Information ---")
        report_lines.append(f"Number of rows: {df.shape[0]}")
        report_lines.append(f"Number of columns: {df.shape[1]}")
        report_lines.append("\nColumn names:")
        for col in df.columns:
            report_lines.append(f"- {col}")

        report_lines.append("\n--- Data Types ---")
        # Capture df.info() output
        import io
        buffer = io.StringIO()
        df.info(buf=buffer)
        report_lines.append(buffer.getvalue())

        report_lines.append("\n--- First 5 Rows (Head) ---")
        report_lines.append(df.head().to_string())

        report_lines.append("\n\n--- Last 5 Rows (Tail) ---")
        report_lines.append(df.tail().to_string())

        report_lines.append("\n\n--- Descriptive Statistics (for numerical columns) ---")
        report_lines.append(df.describe(include='all').to_string())

        report_lines.append("\n\n--- Missing Values per Column ---")
        report_lines.append(df.isnull().sum().to_string())

        report_lines.append("\n\nInspection complete.")
        return "\n".join(report_lines)

    except pd.errors.EmptyDataError:
        return f"Error: The file at {dataset_path} is empty."
    except pd.errors.ParserError:
        return f"Error: The file at {dataset_path} does not appear to be a valid CSV file."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    inspection_report = inspect_dataset(DATASET_PATH)

    # Ensure memory-bank directory exists
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(inspection_report)

    print(f"Inspection report saved to {REPORT_PATH}")
    print("\nPreview of the report:")
    print(inspection_report)
