"""
Pyxon AI — Pull secrets from AWS Secrets Manager and write .env

Usage:
    python setup_env.py

No AWS CLI, no AWS account, no configuration needed.
Credentials are embedded (read-only, scoped to this project's secrets only).
"""

import json
import os
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

# Read-only IAM credentials scoped to secretsmanager:GetSecretValue on this
# single secret. They cannot access any other AWS resource.
AWS_ACCESS_KEY_ID = "AKIAW5PNQJS24XFLAVO7"
AWS_SECRET_ACCESS_KEY = "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"


def main() -> None:
    print("Fetching secrets from AWS Secrets Manager...")

    client = boto3.client(
        "secretsmanager",
        region_name=REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
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
