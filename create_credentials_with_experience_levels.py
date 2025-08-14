import pandas as pd
import random
import re

def slugify(name: str) -> str:
    """Turn textbook name into a safe column suffix."""
    s = name.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)  # non-alphanumerics -> _
    s = re.sub(r"_+", "_", s).strip("_")  # collapse & trim _
    return s

def generate_sample_credentials_csv(output_path="sample_credentials_with_levels.csv"):
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    levels = ["Beginner", "Intermediate", "Advanced"]
    textbooks = [
        "Introductory Macroeconomics",
        "Introductory Microeconomics",
        "Statistics For Economics",
        "MATHEMATICS Textbook for Class IX",
        "MATHEMATICS Textbook for Class X",
        "MATHEMATICS Textbook for Class XI",
        "MATHEMATICS Textbook for Class XII PART I",
        "MATHEMATICS Textbook for Class XII PART II",
    ]
    chat_history_flag = ["yes", "no"]

    # Build dynamic level column names for each textbook
    level_cols = {tb: f"{slugify(tb)}_level" for tb in textbooks}

    rows = []
    for i in range(1, 31):
        username = f"user{i:03d}"
        password = f"user{i:03d}"              
        voice = random.choice(voices)
        assigned_subject = random.choice(textbooks)
        chat_history = random.choice(chat_history_flag)

        # Base row
        row = {
            "username": username,
            "password": password,
            "voice": voice,
            "assigned_subject": assigned_subject,
            "chat_history": chat_history,
        }

        # One level for each textbook (dynamic columns)
        for tb, col in level_cols.items():
            row[col] = random.choice(levels)

        rows.append(row)

    # Ensure a consistent column order: core fields → dynamic level cols 
    core = ["username", "password", "voice", "assigned_subject", "chat_history"]
    dynamic = list(level_cols.values())


    df = pd.DataFrame(rows)[core + dynamic]
    df.to_csv(output_path, index=False)
    print(f"✅ Credentials file saved to: {output_path}")
    print(f"🧩 Columns created: {', '.join(dynamic)}")

# Run it
generate_sample_credentials_csv()

