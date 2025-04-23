import csv
import json
import logging  # Added logging
from io import StringIO
from typing import Any, Dict, List, Optional, TextIO  # Added typing, TextIO, cast

import click  # Import click
import instructor
import openai
from pydantic import BaseModel, Field
from rich.console import Console  # Added rich import
from rich.table import Table  # Added rich import

# --- Configuration ---
JSON_FILE_PATH = "data/dev_20240627/dev_tables.json"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"  # Required, but unused by Ollama
OLLAMA_MODEL = "gemma3:27b-it-qat"
ENDPOINT_TYPE = "ollama"  # Added endpoint type
LOG_LEVEL = logging.INFO  # Configure log level

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s: %(message)s")

# --- Ollama Client Setup ---
try:
    client = instructor.from_openai(
        openai.OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key=OLLAMA_API_KEY,
        ),
        mode=instructor.Mode.JSON,
    )
except Exception as e:
    logging.error(f"Failed to initialize Ollama client: {e}")
    exit(1)


# --- Pydantic Models ---
class TableSchemaInput(BaseModel):
    table_name: str = Field(..., description="The technical name of the table.")
    full_name: str = Field(
        ...,
        description="A more descriptive, human-friendly name for the table's topic.",
    )
    schema_csv: str = Field(
        ...,
        description="CSV data detailing the table's columns: original_column_name,column_name,column_description,data_format,value_description",
    )


class TableSummary(BaseModel):
    summary: str = Field(
        ...,
        description="A comprehensive, human-readable summary of the database table.",
    )


# --- System Prompt ---
SYSTEM_PROMPT = """Create a comprehensive, human-readable summary of a database table. This summary should explain the table's purpose, the type of information it contains, and how the key columns relate to each other, going beyond simple keyword listing.

Input: You will be provided with the following information for a database table:
1.  `table_name:` The technical name of the table.
2.  `full_name:` A more descriptive, human-friendly name for the table's topic.
3.  `---schema.csv` followed by CSV data detailing the table's columns. The CSV columns are: `original_column_name`, `column_name`, `column_description`, `data_format`, `value_description`. Pay close attention to `column_name` (as description) and `data_format` for context.

Task: Based on the provided input, generate a descriptive summary paragraph (or a few short paragraphs if necessary) that explains the table.

Your summary MUST:

1.  State the Purpose: Clearly state the main topic or purpose of the table, using the `full_name` and `table_name` for context.
2.  Describe the Content: Explain what kind of data is stored in the table. What entity does each row typically represent (e.g., a specific school, a student record, a transaction)? Use the `column_name` (human-readable) fields to understand this.
3.  Identify Key Dimensions/Identifiers: Mention the main columns used to identify or categorize the data (e.g., ID codes, names, types, time periods).
4.  Highlight Key Metrics/Values: Describe the important quantitative or qualitative data points the table provides (e.g., counts, ratings, status flags). Explain what these metrics represent based on their names and types.
5.  Synthesize, Don't Just List: Combine the information from the columns into a coherent narrative. Avoid simply listing column names.
6.  Format: Present the summary as clear, well-structured prose.

Now, analyze the following table schema and generate the detailed summary:
"""

# --- Core Functions ---


def load_database_schemas(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """Loads database schemas from a JSON file."""
    try:
        with open(file_path, "r") as f:
            databases = json.load(f)
        logging.info(f"Successfully loaded database schemas from {file_path}")
        return databases
    except FileNotFoundError:
        logging.error(f"JSON file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading {file_path}: {e}")
        return None


def create_schema_csv(
    table_index: int,
    cols_orig: List[List[Any]],
    cols_friendly: List[List[Any]],
    col_types: List[str],
) -> Optional[str]:
    """Creates a CSV string representing the schema for a specific table."""
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "original_column_name",
            "column_name",
            "column_description",
            "data_format",
            "value_description",
        ]
    )
    columns_processed_count = 0

    for col_idx, col_orig_pair in enumerate(cols_orig):
        if col_idx >= len(cols_friendly) or col_idx >= len(col_types):
            logging.warning(
                f"Column index {col_idx} out of bounds for table {table_index}. Skipping remaining columns."
            )
            break

        if col_orig_pair[0] != table_index:
            continue  # Skip columns not belonging to the current table

        original_name = col_orig_pair[1]
        if original_name == "*":  # Skip wildcard column
            continue

        friendly_name_pair = cols_friendly[col_idx]
        if friendly_name_pair[0] != table_index:
            logging.warning(
                f"Mismatch in table index between original ({table_index}) and friendly ({friendly_name_pair[0]}) column data at index {col_idx}. Skipping column."
            )
            continue

        friendly_name = friendly_name_pair[1]
        col_type = col_types[col_idx]
        description = (
            friendly_name  # Using friendly name as description per prompt refinement
        )
        value_desc = ""  # Placeholder

        writer.writerow(
            [original_name, friendly_name, description, col_type, value_desc]
        )
        columns_processed_count += 1

    if columns_processed_count == 0:
        logging.warning(
            f"No valid columns found or processed for table index {table_index}. Cannot create CSV."
        )
        output.close()
        return None

    schema_csv_str = output.getvalue()
    output.close()
    return schema_csv_str


def get_table_summary(table_input: TableSchemaInput) -> Optional[TableSummary]:
    """Generates a human-readable summary for a given table schema input using an LLM."""
    formatted_input_str = f"""table_name: {table_input.table_name}
full_name: {table_input.full_name}
---schema.csv
{table_input.schema_csv}"""

    try:
        # Make the request potentially longer timeout if needed for larger models
        response = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": formatted_input_str},
            ],
            response_model=TableSummary,
            max_retries=2,  # Optional: Add retries
            # timeout=120 # Optional: Increase timeout if needed
        )
        return response  # type: ignore
    except Exception as e:
        logging.error(f"LLM API call failed for table '{table_input.table_name}': {e}")
        return None


def process_table(
    db_id: str,  # Add db_id
    table_name_orig: str,
    table_name_friendly: str,
    schema_csv: str,
    output_file: TextIO,  # Output file handle (for JSONL)
    total_tables: int,  # Add total table count
    processed_tables_tracker: List[int],  # Add tracker
) -> Dict[str, Any]:  # Modified return type
    """Processes a single table: creates input, calls LLM, writes JSON line, updates progress, and returns status."""
    logging.info(
        f"  Generating summary for table: {table_name_orig} ({table_name_friendly}) in DB: {db_id}"
    )

    output_record: Dict[str, Any] = {
        "db_id": db_id,
        "table_name_original": table_name_orig,
        "table_name_friendly": table_name_friendly,
        "schema_csv": schema_csv,
        "summary": None,
        "error": None,
        "model_name": OLLAMA_MODEL,
        "endpoint_type": ENDPOINT_TYPE,
    }

    table_input = TableSchemaInput(
        table_name=table_name_orig,
        full_name=table_name_friendly,
        schema_csv=schema_csv,
    )

    summary_output = get_table_summary(table_input)
    if summary_output:
        # Actions for success
        output_record["summary"] = summary_output.summary
        logging.info(
            f"    Successfully generated summary for {table_name_orig}."
        )  # Simplified log slightly
        status = "Success"
        details = (
            output_record["summary"][:75] + "..."
            if output_record["summary"] and len(output_record["summary"]) > 75
            else output_record["summary"]
        )
    else:
        # Actions for failure
        error_msg = f"Failed to generate summary for table {table_name_orig}."
        output_record["error"] = error_msg
        logging.error(f"    {error_msg}")
        status = "Failure"
        details = error_msg

    # Common actions (write to file, update progress, return)
    try:
        json_line = json.dumps(output_record)
        output_file.write(json_line + "\\n")
    except (TypeError, OverflowError) as json_err:
        logging.error(
            f"    Failed to serialize record to JSON for table {table_name_orig}: {json_err}"
        )

    processed_tables_tracker[0] += 1
    current_count = processed_tables_tracker[0]
    logging.info(f"  Progress: {current_count}/{total_tables} tables processed.")

    return {
        "db_id": db_id,
        "table_name": table_name_orig,
        "status": status,
        "details": details,
    }


def process_database(
    db_schema: Dict[str, Any],
    output_file: TextIO,
    total_tables: int,  # Add total table count
    processed_tables_tracker: List[int],  # Add tracker
) -> List[Dict[str, Any]]:  # Modified return type
    """Processes all tables within a single database schema."""
    db_id = db_schema.get("db_id", "Unknown DB")
    logging.info(f"--- Processing Database: {db_id} ---")

    tables_orig = db_schema.get("table_names_original", [])
    tables_friendly = db_schema.get("table_names", [])
    cols_orig = db_schema.get("column_names_original", [])
    cols_friendly = db_schema.get("column_names", [])
    col_types = db_schema.get("column_types", [])

    if len(tables_orig) != len(tables_friendly):
        logging.warning(
            f"Mismatch between original ({len(tables_orig)}) and friendly ({len(tables_friendly)}) table name counts for DB '{db_id}'. Skipping DB."
        )
        return []

    results = []  # List to store results for this DB
    for tbl_idx, table_name_orig in enumerate(tables_orig):
        table_name_friendly = tables_friendly[tbl_idx]

        schema_csv_str = create_schema_csv(tbl_idx, cols_orig, cols_friendly, col_types)

        if schema_csv_str:
            result = process_table(
                db_id,  # Pass db_id
                table_name_orig,
                table_name_friendly,
                schema_csv_str,
                output_file,
                total_tables,  # Pass total
                processed_tables_tracker,  # Pass tracker
            )  # Pass handle
            results.append(result)  # Collect result
        else:
            logging.warning(
                f"Skipping summary generation for table '{table_name_orig}' in DB '{db_id}' due to CSV creation issues."
            )
            # Add error result for skipped table
            results.append(
                {
                    "db_id": db_id,
                    "table_name": table_name_orig,
                    "status": "Failure",
                    "details": f"Could not create schema CSV for table '{table_name_orig}'.",
                }
            )
            # Also update progress even if skipped/failed at CSV stage
            processed_tables_tracker[0] += 1
            current_count = processed_tables_tracker[0]
            logging.warning(
                f"  Progress: {current_count}/{total_tables} tables processed (CSV failed)."
            )

    return results  # Return results for this DB


# --- Click Command Definition ---
@click.command()
@click.option(
    "--input-json",
    default=JSON_FILE_PATH,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help=f"Path to the input JSON file containing database schemas. Default: {JSON_FILE_PATH}",
)
@click.option(
    "-o",
    "--output-file",
    required=True,
    type=click.Path(dir_okay=False, writable=True),
    help="Path to the output JSONL file where summaries will be written.",
)
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Set the logging level.",
)
def main(input_json: str, output_file: str, log_level: str):
    """
    Generates human-readable summaries for database tables defined in a JSON file
    and writes them to an output file. Displays a summary table at the end.
    """
    # --- Reconfigure Logging based on CLI arg ---
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(levelname)s: %(message)s",
        force=True,  # Override basicConfig called earlier
    )

    logging.info("Starting table summary generation process...")
    databases = load_database_schemas(input_json)  # Use argument

    if databases:
        # Calculate total number of tables
        total_tables = 0
        for db_schema in databases:
            total_tables += len(db_schema.get("table_names_original", []))
        logging.info(
            f"Found {total_tables} tables to process across {len(databases)} databases."
        )

        processed_tables_tracker = [0]  # List to pass counter by reference
        all_results = []  # List to store all results

        try:
            # Open the output file
            with open(output_file, "w", encoding="utf-8") as outfile:
                logging.info(f"Writing summaries to: {output_file}")
                for db_schema in databases:
                    db_results = process_database(
                        db_schema, outfile, total_tables, processed_tables_tracker
                    )  # Pass counter and total
                    all_results.extend(db_results)  # Collect results

                    # --- Print Summary Table for the Current DB ---
                    if db_results:
                        db_id = db_schema.get(
                            "db_id", "Unknown DB"
                        )  # Get DB ID for title
                        console = Console()
                        table = Table(
                            show_header=True,
                            header_style="bold cyan",  # Slightly different style for intermediate tables
                            title=f"Processing Summary for DB: {db_id}",
                        )
                        table.add_column("Table Name", min_width=20)
                        table.add_column("Status", justify="center")
                        table.add_column("Details / Summary Snippet", max_width=80)

                        for result in db_results:
                            status_style = (
                                "green" if result["status"] == "Success" else "red"
                            )
                            table.add_row(
                                result["table_name"],
                                f"[{status_style}]{result['status']}[/]",
                                result["details"] or "N/A",
                            )
                        console.print(table)
                    # --- End Intermediate Table Print ---

                logging.info("Table summary generation process finished.")

            # --- Print Final Summary Table ---
            if all_results:
                console = Console()
                table = Table(
                    show_header=True,
                    header_style="bold magenta",
                    title="Processing Summary",
                )
                table.add_column("DB ID", style="dim", width=15)
                table.add_column("Table Name", min_width=20)
                table.add_column("Status", justify="center")
                table.add_column("Details / Summary Snippet", max_width=80)

                for result in all_results:
                    status_style = "green" if result["status"] == "Success" else "red"
                    table.add_row(
                        result["db_id"],
                        result["table_name"],
                        f"[{status_style}]{result['status']}[/]",
                        result["details"]
                        or "N/A",  # Handle cases where details might be None
                    )

                console.print(table)
                logging.info(
                    "Final processing summary table displayed."
                )  # Added log message
            else:
                logging.info("No tables were processed or found.")

        except IOError as e:
            logging.error(f"Could not open or write to output file {output_file}: {e}")
            exit(1)  # Exit if file cannot be opened/written
    else:
        logging.error(f"Could not load database schemas from {input_json}. Exiting.")
        exit(1)


# --- Main Execution ---
if __name__ == "__main__":
    main()  # Call the click-decorated main function
