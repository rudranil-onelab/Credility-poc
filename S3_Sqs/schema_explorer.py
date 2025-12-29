import pandas as pd
import re
from datetime import datetime
import pandas as pd

class SchemaExplorer:
    def __init__(self, conn, db_type):
        self.conn = conn
        self.db_type = db_type
        self.tables = self._load_schema()

    def _load_schema(self):

        tables = {}

        # ---------- MYSQL ----------
        if self.db_type == "mysql":
            df = pd.read_sql("""
                SELECT 
                    table_name  AS table_name,
                    column_name AS column_name,
                    data_type   AS data_type
                FROM information_schema.columns
                WHERE table_schema = DATABASE()
            """, self.conn)

        # ---------- POSTGRES ----------
        elif self.db_type == "postgres":
            df = pd.read_sql("""
                SELECT 
                    table_name  AS table_name,
                    column_name AS column_name,
                    data_type   AS data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
            """, self.conn)

        # ---------- SQLITE ----------
        elif self.db_type == "sqlite":
            df = None
            c = self.conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table'")
            for (t,) in c.fetchall():
                c.execute(f"PRAGMA table_info({t})")
                for col in c.fetchall():
                    tables.setdefault(t, []).append({
                        "column_name": col[1],
                        "data_type": col[2]
                    })
            return tables

        # ---------- SNOWFLAKE ----------
        elif self.db_type == "snowflake":
            cur = self.conn.cursor()
            cur.execute("""
                SELECT 
                    table_name  AS "table_name",
                    column_name AS "column_name",
                    data_type   AS "data_type"
                FROM information_schema.columns
                WHERE table_schema = CURRENT_SCHEMA()
            """)
            df = pd.DataFrame(cur.fetchall(), columns=["table_name","column_name","data_type"])
            cur.close()

        else:
            raise ValueError("Unsupported DB")

        # normalize column names
        df.columns = [c.lower() for c in df.columns]

        for _, r in df.iterrows():
            tables.setdefault(r["table_name"], []).append({
                "column_name": r["column_name"],
                "data_type": r["data_type"]
            })

        return tables


    def query(self, sql):
        df = pd.read_sql(sql, self.conn)
        return df.to_dict(orient="records")

    def normalize(value):
        if value is None:
            return ""

        val = str(value).strip().lower()

        # Remove spaces and non-alphanumerics (for Aadhaar, PAN, etc)
        val = re.sub(r"[^a-z0-9]", "", val)

        # Try date normalization
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(str(value), fmt)
                return dt.strftime("%Y%m%d")
            except:
                pass

        return val