import os
import pandas as pd

from src.features.minimal_extractor import MinimalFeatureExtractor

# Folder containing all datasets
DATASET_FOLDER = "datasets"

# Generated training dataset
OUTPUT_FILE = "datasets/training_dataset.csv"


def infer_label(column_name, series):
    """
    Generate preprocessing label for each column.
    These labels are used to train the ML models.
    """

    name = column_name.lower()

    # High missing values
    if series.isnull().mean() > 0.30:
        return "drop_column"

    # ID-like columns
    if any(x in name for x in ["id", "uuid", "index"]):
        return "drop_column"

    # Boolean columns
    if pd.api.types.is_bool_dtype(series):
        return "parse_boolean"

    # Datetime columns
    if pd.api.types.is_datetime64_any_dtype(series):
        return "keep_as_is"

    # Numeric columns
    if pd.api.types.is_numeric_dtype(series):

        if series.dropna().empty:
            return "keep_as_is"

        try:
            if abs(series.skew()) > 1:
                return "log1p_transform"
        except Exception:
            pass

        return "standard_scale"

    # Object/String columns
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):

        unique_ratio = series.nunique(dropna=True) / max(len(series), 1)

        if unique_ratio < 0.30:
            return "label_encode"

        return "text_clean"

    return "keep_as_is"


def main():

    extractor = MinimalFeatureExtractor()

    rows = []

    print("=" * 60)
    print("GENERATING TRAINING DATASET")
    print("=" * 60)

    for filename in os.listdir(DATASET_FOLDER):

        if not filename.endswith(".csv"):
            continue

        if filename == "training_dataset.csv":
            continue

        filepath = os.path.join(DATASET_FOLDER, filename)

        print(f"\nProcessing: {filename}")

        try:

            df = pd.read_csv(filepath)

            for column_name in df.columns:

                column = df[column_name]

                features = extractor.extract(
                    column=column,
                    column_name=column_name
                )

                row = features.to_dict()

                row["dataset"] = filename
                row["column_name"] = column_name
                row["label"] = infer_label(column_name, column)

                rows.append(row)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    training_df = pd.DataFrame(rows)

    os.makedirs(DATASET_FOLDER, exist_ok=True)

    training_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("TRAINING DATASET CREATED SUCCESSFULLY")
    print("=" * 60)

    print(f"Total Samples : {len(training_df)}")
    print(f"Total Features: {training_df.shape[1]}")
    print(f"Saved To      : {OUTPUT_FILE}")

    print("\nPreview:")
    print(training_df.head())


if __name__ == "__main__":
    main()