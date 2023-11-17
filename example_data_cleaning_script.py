import pandas as pd
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table

def load_data(file_path):
    """ Load data from a CSV file """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        console.print(f"[red]Error loading file: {e}[/red]")
        return None

def display_data(df, console, rows=5):
    """ Display the first few rows of the dataframe in a table format """
    table = Table(show_header=True, header_style="bold magenta")
    for col in df.columns:
        table.add_column(col, style="dim")
    for _, row in df.head(rows).iterrows():
        table.add_row(*[str(val) for val in row])
    console.print(table)

def confirm_name_and_member(df, console, output_path):
    """ Confirm if 'name' and 'committee_member' are the same """
    for index, row in df.iterrows():
        if row['name'] != row['committee_member']:
            console.print(f"[yellow]Name and Committee Member do not match for record {index}[/yellow]")
            console.print(f"Name: {row['name']}, Committee Member: {row['committee_member']}")
            if Confirm.ask("Do you want to correct this?"):
                corrected_name = Prompt.ask("Enter the correct name")
                df.at[index, 'name'] = corrected_name
                df.at[index, 'committee_member'] = corrected_name
                df.to_csv(output_path, index=False)
    return df

def add_research_areas(df, console, output_path):
    """ Add additional research areas """
    for index, row in df.iterrows():
        console.print(f"Current Research Area for {row['name']}: {row['research_area']}")
        if Confirm.ask("Do you want to add more research areas?"):
            additional_areas = Prompt.ask("Enter additional research areas, separated by commas")
            df.at[index, 'research_area'] += ", " + additional_areas
            df.to_csv(output_path, index=False)
    return df

# Main Execution
if __name__ == '__main__':
    console = Console()

    console.print("[yellow]Data Cleaning CLI Application[/yellow]")
    file_path = "scraped_ischool_people.csv"
    df = load_data(file_path)

    if df is not None:
        console.print("[green]Data loaded successfully[/green]")
        display_data(df, console)

        df = confirm_name_and_member(df, console)
        df = add_research_areas(df, console)
