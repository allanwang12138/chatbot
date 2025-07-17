import pandas as pd
import random

def generate_sample_credentials_csv(output_path="sample_credentials_with_levels.csv"):
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    levels = ["Beginner", "Intermediate", "Advanced"]

    data = []
    for i in range(1, 31):
        username = f"user{i:03d}"
        password = f"macro{i:03d}"
        voice = random.choice(voices)
        macro_level = random.choice(levels)
        micro_level = random.choice(levels)
        physics_level = random.choice(levels)

        data.append({
            "username": username,
            "password": password,
            "voice": voice,
            "macro_level": macro_level,
            "micro_level": micro_level,
            "physics_level": physics_level
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✅ Credentials file saved to: {output_path}")

# Call the function
generate_sample_credentials_csv()
