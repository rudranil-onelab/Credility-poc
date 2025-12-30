import os

CLIENT_DB_REGISTRY = {
    "client_123": {
        "db_type": "mysql",
        "host": os.getenv("CLIENT1_DB_HOST"),
        "port": 3306,
        "database": os.getenv("CLIENT1_DB_NAME"),
        "user": os.getenv("CLIENT1_DB_USER"),
        "password": os.getenv("CLIENT1_DB_PASSWORD"),
    },

    "client_456": {
        "db_type": "postgres",
        "host": os.getenv("CLIENT2_DB_HOST"),
        "port": 5432,
        "database": os.getenv("CLIENT2_DB_NAME"),
        "user": os.getenv("CLIENT2_DB_USER"),
        "password": os.getenv("CLIENT2_DB_PASSWORD"),
    },

    "client_789": {
        "db_type": "sqlite",
        "db_path": os.getenv("CLIENT3_DB_PATH"),
    },

    "client_analytics": {
        "db_type": "snowflake",
        "account": os.getenv("SF_ACCOUNT"),
        "user": os.getenv("SF_USER"),
        "password": os.getenv("SF_PASSWORD"),
        "warehouse": os.getenv("SF_WAREHOUSE"),
        "database": os.getenv("SF_DATABASE"),
        "schema": "PUBLIC",
    },
}
