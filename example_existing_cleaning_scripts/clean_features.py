import pandas as pd
from rich.console import Console
from rich.table import Table
import numpy as np
import os
import ast

def safe_literal_eval(val):
    """
    Safely evaluate an expression node or a string containing a Python expression.
    Returns the original value if an error occurs during evaluation.
    """
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

def rename_columns(df, prefix):
    """
    Rename columns of a dataframe except for 'column_name'.
    """
    cols = df.columns.tolist()
    cols.remove('column_name')
    df.columns = ['column_name'] + [prefix + col for col in cols]
    return df

def load_and_prepare_data(output_path, tw_path, sampled_path):
    """
    Load, merge, and prepare datasets.
    """
    if os.path.exists(output_path):
        return pd.read_csv(output_path)
    
    tw_df = pd.read_csv(tw_path)
    sampled_df = pd.read_csv(sampled_path)

    # Rename columns to avoid conflicts after merging
    tw_df = rename_columns(tw_df, 'tw_')
    sampled_df = rename_columns(sampled_df, 'serials_')

    combined_df = pd.merge(tw_df, sampled_df, on='column_name', how='outer')
    subset_combined_column_distribution_df = combined_df[(combined_df.tw_unique_values.notna()) & (combined_df.serials_unique_values.notna())]
    subset_combined_column_distribution_df.to_csv(output_path, index=False)
    return subset_combined_column_distribution_df

def get_feature_type(row, console, categories):
    """
    Get the feature type based on user input.
    """
    input = console.input("Which category to use? (tw/serials) (default: {}, otherwise {}). OR press enter to ".format(row.tw_category, row.serials_category))
    if input == "tw":
        return row.tw_category
    elif input == "serials":
        return row.serials_category
    elif input == "":
        return choose_from_categories(console, categories)
    return None

def choose_from_categories(console, categories):
    """
    Allow the user to choose from predefined categories.
    """
    input = console.input("Select the correct feature type: {}. Press 1, 2, or 3 for the corresponding category. ".format(", ".join(categories)))
    return categories[int(input) - 1] if input in ["1", "2", "3"] else None

def user_input_for_classification(row, console, categories):
    """
    Obtain user input for feature classification.
    """
    keep_feature = console.input("Is this a feature? (y/n) ") == "y"
    feature_type = None
    if keep_feature:
        feature_type = get_feature_type(row, console, categories)
    return keep_feature, feature_type

def print_category_counts(df, classification, col, category, console):
    """
    Print value counts for a specific classification and category in a console table.
    """
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Value", style="dim", width=12)
    table.add_column("Count", justify="right")

    for value, count in df[df.classification == classification][col].value_counts().items():
        table.add_row(str(value), str(count))

    console.print("\n{} corpus, not null: {} and category: {}".format(
        classification, df[df.classification == classification][col].notna().sum(), category))
    console.print(table)

def classify_features(mismatch_df, full_df, mapped_df, console, categories, output_path, subset_combined_column_distribution_df):
    """
    Classify features based on user input.
    """
    for index, row in mismatch_df.iterrows():
        console.print("\nColumn: {} number {} out of {}".format(row.column_name, index, len(mismatch_df)))
        actual_col = mapped_df[mapped_df.cleaned_column_name == row.column_name].column_name.values[0]
        print_category_counts(full_df, 'third_world_serials', actual_col, row.tw_category, console)
        print_category_counts(full_df, 'sampled_serials', actual_col, row.serials_category, console)
        
        keep_feature, feature_type = user_input_for_classification(row, console, categories)
        subset_combined_column_distribution_df.loc[subset_combined_column_distribution_df.column_name == row.column_name, 'keep_feature'] = keep_feature
        subset_combined_column_distribution_df.loc[subset_combined_column_distribution_df.column_name == row.column_name, 'feature_type'] = feature_type
        
        subset_combined_column_distribution_df.to_csv(output_path, index=False)







# Main Execution
if __name__ == '__main__':
    console = Console()
    output_path = "../datasets/combined_column_distribution.csv"
    tw_path = "../datasets/tw_column_distribution.csv"
    sampled_path = "../datasets/serials_column_distribution.csv"
    subset_combined_column_distribution_df = load_and_prepare_data(output_path, tw_path, sampled_path)

    mismatch_df = subset_combined_column_distribution_df[subset_combined_column_distribution_df.tw_category != subset_combined_column_distribution_df.serials_category][['column_name', 'tw_category', 'serials_category']]

    full_df = pd.read_csv("../datasets/combined_classified_serials_dataset.csv")
    full_df = full_df.applymap(safe_literal_eval)

    # Then replace empty strings with None
    full_df = full_df.replace('', None)
    full_df = full_df.replace(' ', None)
    mapped_df = pd.read_csv("../datasets/marc_column_mapping.csv")

    categories = ["well suited for categorical", "could be categorical", "too many for categorical"]

    if 'keep_feature' not in subset_combined_column_distribution_df.columns:
        subset_combined_column_distribution_df['keep_feature'] = None

    if 'feature_type' not in subset_combined_column_distribution_df.columns:
        subset_combined_column_distribution_df['feature_type'] = None

    classify_features(mismatch_df, full_df, mapped_df, console, categories, output_path, subset_combined_column_distribution_df)