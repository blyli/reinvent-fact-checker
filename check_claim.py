import csv
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from langsmith.wrappers import wrap_openai  # noqa: E402
from openai import OpenAI  # noqa: E402

CSV_PATH = ROOT / "data" / "aws_reinvent_candidate_items.csv"
ROW_ID = "C-02"
QUESTION = "Is this a real AWS launch? Answer yes or no, and name the page that proves it."


def load_row(path: Path, row_id: str) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["candidate_id"] == row_id:
                return row
    raise SystemExit(f"{row_id} not found in {path}")


def build_prompt(row: dict) -> str:
    return (
        f"Claim: {row['claim_text']}\n"
        f"Service: {row['claimed_service']}\n"
        f"Claim type: {row['claim_type']}\n"
        f"Source name: {row['origin_source_name']}\n"
        f"Source URL: {row['origin_source_url']}\n"
        f"Source type: {row['origin_source_type']}\n"
        f"Published: {row['origin_published_date']}\n\n"
        f"{QUESTION}"
    )


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("OPENROUTER_MODEL_ID")
    if not api_key or not model:
        raise SystemExit("OPENROUTER_API_KEY and OPENROUTER_MODEL_ID must be set in .env")

    row = load_row(CSV_PATH, ROW_ID)
    client = wrap_openai(OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key))
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": build_prompt(row)}],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
