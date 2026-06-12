import pandas as pd
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import filedialog


def clean_features(input_csv, output_csv=None, threshold=0.3):
    """
    Remove rows where the class and corresponding _like feature don't match.
    
    Removes windows (rows) where the _like feature for the predicted class is below threshold.
    For example, if class = "Drops" and drop_like = 0.1 (below 0.3), the row is removed.
    
    Special handling for "Down" class: requires lift_like to be negative (< -threshold).
    This is because lift_like ranges from -1 to 1, where negative values indicate downward motion.
    
    Parameters:
    - input_csv: Path to input CSV file with features per window
    - output_csv: Path to output CSV file (default: adds '_cleaned' suffix)
    - threshold: Minimum confidence threshold for _like features (default: 0.3)
    """
    
    # Read the CSV file
    df = pd.read_csv(input_csv)
    print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    
    # If output file not specified, create default name
    if output_csv is None:
        input_path = Path(input_csv)
        output_csv = input_path.parent / f"{input_path.stem}_cleaned.csv"
    
    # Find the class column
    class_col = None
    for col in df.columns:
        if col.lower() == 'class':
            class_col = col
            break
    
    if class_col is None:
        raise ValueError("No 'class' column found in CSV")
    
    print(f"Found class column: '{class_col}'")
    
    # Get all _like columns
    like_columns = {col: col for col in df.columns if col.lower().endswith('_like')}
    print(f"Found {len(like_columns)} _like columns: {list(like_columns.keys())}")
    
    # Track rows to keep and statistics per class
    rows_to_keep = []
    removed_count = 0
    low_confidence_count = 0
    no_match_count = 0
    class_stats = {}  # {class_name: {'total': int, 'kept': int}}
    
    for idx, row in df.iterrows():
        class_value = row[class_col]
        
        if pd.isna(class_value):
            print(f"Warning: NaN value in class column at row {idx}, removing")
            removed_count += 1
            continue
        
        class_value_str = str(class_value).strip()
        
        # Initialize class stats
        if class_value_str not in class_stats:
            class_stats[class_value_str] = {'total': 0, 'kept': 0}
        class_stats[class_value_str]['total'] += 1
        
        class_value_lower = class_value_str.lower()
        
        # Find the corresponding _like column
        like_col = None
        for col in like_columns.keys():
            col_lower = col.lower()
            # Remove '_like' suffix to get feature name
            feature_name = col_lower[:-5]  # Remove '_like'
            
            # Check if feature name matches class value (handle plurals and case)
            if feature_name == class_value_lower or \
               feature_name.rstrip('s') == class_value_lower.rstrip('s'):
                like_col = col
                break
        
        if like_col is None:
            print(f"Warning: No matching _like column found for class '{class_value}' at row {idx}")
            rows_to_keep.append(idx)
            class_stats[class_value_str]['kept'] += 1
            no_match_count += 1
            continue
        
        # Check if _like value meets the confidence threshold
        like_value = row[like_col]
        
        if pd.isna(like_value):
            print(f"Warning: NaN value in {like_col} at row {idx}, removing")
            removed_count += 1
            low_confidence_count += 1
            continue
        
        try:
            like_value = float(like_value)
        except (ValueError, TypeError):
            print(f"Warning: Could not convert {like_col} value '{like_value}' to float at row {idx}, removing")
            removed_count += 1
            low_confidence_count += 1
            continue
        
        # Special handling for "Down" class: lift_like should be negative
        if class_value_lower == "down" or class_value_lower == "downs":
            # For Down, we expect lift_like to be negative (less than -threshold)
            if like_value < -threshold:
                rows_to_keep.append(idx)
                class_stats[class_value_str]['kept'] += 1
            else:
                removed_count += 1
                low_confidence_count += 1
        else:
            # For other classes, require _like value >= threshold
            if like_value >= threshold:
                rows_to_keep.append(idx)
                class_stats[class_value_str]['kept'] += 1
            else:
                removed_count += 1
                low_confidence_count += 1
    
    # Create cleaned dataframe
    df_cleaned = df.loc[rows_to_keep].reset_index(drop=True)
    
    # Save to CSV
    df_cleaned.to_csv(output_csv, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE CLEANUP SUMMARY")
    print("="*60)
    print(f"Input file:           {input_csv}")
    print(f"Output file:          {output_csv}")
    print(f"Confidence threshold: {threshold}")
    print("-"*60)
    print(f"Original rows:        {len(df)}")
    print(f"Rows removed:         {removed_count}")
    print(f"  - Low confidence:   {low_confidence_count}")
    print(f"  - No match found:   {no_match_count}")
    print(f"Rows retained:        {len(df_cleaned)}")
    print(f"Retention rate:       {100*len(df_cleaned)/len(df):.1f}%")
    print("-"*60)
    print("RETENTION PER CLASS:")
    print("-"*60)
    
    # Sort classes by name for consistent output
    for class_name in sorted(class_stats.keys()):
        stats = class_stats[class_name]
        if stats['total'] > 0:
            retention_pct = 100 * stats['kept'] / stats['total']
            print(f"  {class_name:20} {stats['kept']:4}/{stats['total']:4} kept ({retention_pct:5.1f}%)")
    
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean feature CSV by removing rows where class _like feature is below threshold"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        nargs="?",
        default=None,
        help="Path to input CSV file with features per window"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to output CSV file (default: input_file_cleaned.csv)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.3,
        help="Minimum confidence threshold for _like features (default: 0.3)"
    )
    
    args = parser.parse_args()
    
    # If no input file provided, show file dialog
    if args.input_csv is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        input_csv = filedialog.askopenfilename(
            title="Select input CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not input_csv:
            print("No file selected. Exiting.")
            exit(0)
        
        args.input_csv = input_csv
        
        # Optionally ask for output file
        root.withdraw()
        output_csv = filedialog.asksaveasfilename(
            title="Save cleaned CSV as",
            defaultextension=".csv",
            initialfile=Path(input_csv).stem + "_cleaned.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if output_csv:
            args.output = output_csv
    
    try:
        clean_features(args.input_csv, args.output, args.threshold)
        print("\n✓ Feature cleanup completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during feature cleanup: {e}")
        import traceback
        traceback.print_exc()
