import mysql.connector
import psycopg2
import sqlite3
import snowflake.connector
from fastapi import HTTPException, status
from S3_Sqs.client_registry import CLIENT_DB_REGISTRY


def get_client_database_connection(client_id: str):
    """Get client-specific database connection (MySQL, Postgres, SQLite, Snowflake)"""

    try:
        if client_id not in CLIENT_DB_REGISTRY:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invalid client_id"
            )

        cfg = CLIENT_DB_REGISTRY[client_id]
        db_type = cfg["db_type"]

        # ---------- MYSQL ----------
        if db_type == "mysql":
            conn = mysql.connector.connect(
                host=cfg["host"],
                port=int(cfg["port"]),
                user=cfg["user"],
                password=cfg["password"],
                database=cfg["database"],
                autocommit=True
            )
            if not conn.is_connected():
                raise Exception("MySQL connection failed")
            return conn

        # ---------- POSTGRES ----------
        elif db_type == "postgres":
            print("Connecting to Postgres")
            return psycopg2.connect(
                host=cfg["host"],
                port=int(cfg["port"]),
                user=cfg["user"],
                password=cfg["password"],
                dbname=cfg["database"]
            )

        # ---------- SQLITE ----------
        elif db_type == "sqlite":
            return sqlite3.connect(cfg["db_path"], check_same_thread=False)

        # ---------- SNOWFLAKE ----------
        elif db_type == "snowflake":
            return snowflake.connector.connect(
                account=cfg["account"],
                user=cfg["user"],
                password=cfg["password"],
                warehouse=cfg["warehouse"],
                database=cfg["database"],
                schema=cfg.get("schema", "PUBLIC"),
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported db_type: {db_type}"
            )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {str(e)}"
        )
