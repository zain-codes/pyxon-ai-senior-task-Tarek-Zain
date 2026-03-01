"""
Pyxon AI — Pull secrets from AWS Secrets Manager and write .env

Usage:
    1. Place the .env.local file (provided separately) in the project root.
    2. Run: python setup_env.py

The script reads AWS credentials from .env.local, fetches the application
secrets from AWS Secrets Manager, and writes them to .env.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import boto3
except ImportError:
    print("Installing boto3...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "boto3"],
        stdout=subprocess.DEVNULL,
    )
    import boto3

SECRET_NAME = "pyxon-ai/api-keys"
REGION = "us-east-1"


def _load_env_local() -> dict[str, str]:
    """Read key=value pairs from .env.local."""
    env_local = Path(".env.local")
    if not env_local.exists():
        print("Error: .env.local not found.")
        print("")
        print("This file contains the AWS credentials needed to fetch secrets.")
        print("It should have been provided to you separately. Place it in the")
        print("project root and re-run this script.")
        print("")
        print("Expected contents:")
        print("  AWS_ACCESS_KEY_ID=<value>")
        print("  AWS_SECRET_ACCESS_KEY=<value>")
        sys.exit(1)

    values = {}
    for line in env_local.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def main() -> None:
    aws_creds = _load_env_local()

    access_key = aws_creds.get("AWS_ACCESS_KEY_ID", "")
    secret_key = aws_creds.get("AWS_SECRET_ACCESS_KEY", "")

    if not access_key or not secret_key:
        print("Error: .env.local must contain AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
        sys.exit(1)

    print("Fetching secrets from AWS Secrets Manager...")

    client = boto3.client(
        "secretsmanager",
        region_name=REGION,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    try:
        response = client.get_secret_value(SecretId=SECRET_NAME)
    except Exception as exc:
        print(f"\nError: Failed to fetch secret — {exc}")
        sys.exit(1)

    secrets = json.loads(response["SecretString"])

    groq_key = secrets.get("GROQ_API_KEY", "")
    tavily_key = secrets.get("TAVILY_API_KEY", "")

    if not groq_key or not tavily_key:
        print("Error: Secret is missing GROQ_API_KEY or TAVILY_API_KEY.")
        sys.exit(1)

    # Build .env from template (or minimal fallback)
    env_path = Path(".env")
    template = Path(".env.example")

    if template.exists():
        content = template.read_text()
    else:
        content = (
            "GROQ_API_KEY=\n"
            "TAVILY_API_KEY=\n"
            "MODEL_NAME=llama-3.3-70b-versatile\n"
            "SEARCH_MAX_RESULTS=5\n"
            "URL_FETCH_TIMEOUT=10\n"
            "URL_FETCH_MAX_SIZE=51200\n"
            "RAG_ENABLED=true\n"
            "EMBEDDING_MODEL=all-MiniLM-L6-v2\n"
            "LOG_LEVEL=INFO\n"
        )

    content = re.sub(r"^GROQ_API_KEY=.*$", f"GROQ_API_KEY={groq_key}", content, flags=re.MULTILINE)
    content = re.sub(r"^TAVILY_API_KEY=.*$", f"TAVILY_API_KEY={tavily_key}", content, flags=re.MULTILINE)

    env_path.write_text(content)

    print("\nDone! .env file created with secrets from AWS Secrets Manager.\n")
    print("Next steps:")
    print("  docker compose build")
    print("  docker compose run agent python main.py")


if __name__ == "__main__":
    main()
