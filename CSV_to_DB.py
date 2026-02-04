import pandas as pd
import sqlite3
import sys

def convert_to_db(input_file, db_file, table_name="data"):
    # Detect file type
    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
    elif input_file.endswith(".tsv"):
        df = pd.read_csv(input_file, sep="\t")
    elif input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    elif input_file.endswith(".ods"):
        df = pd.read_excel(input_file, engine="odf")
    else:
        raise ValueError("Unsupported file format")

    # Write to SQLite
    conn = sqlite3.connect(db_file)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

# Usage:
# python convert.py input.csv output.db
if __name__ == "__main__":
    convert_to_db(sys.argv[1], sys.argv[2])