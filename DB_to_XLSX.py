import sqlite3
import pandas as pd

conn = sqlite3.connect(r"E:\Aarav PC Data\Code\WHSDSC\whl_2025.db")

tables = pd.read_sql(
    "SELECT name FROM sqlite_master WHERE type='table'",
    conn
)

with pd.ExcelWriter(
    r"E:\Aarav PC Data\Code\WHSDSC\whl_2025_phase_1a.xlsx",
    engine="openpyxl"
) as writer:
    for table in tables["name"]:
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        df.to_excel(writer, sheet_name=table, index=False)

conn.close()
