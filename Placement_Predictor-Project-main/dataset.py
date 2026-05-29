import pandas as pd
import random

# Generate synthetic dataset
data = {
    "IQ": [random.randint(80, 150) for _ in range(100)],
    "CGPA": [round(random.uniform(5.0, 10.0), 2) for _ in range(100)],
}

# Placement logic
data["Placed"] = [
    1 if (iq > 110 and cgpa > 7.5) else 0
    for iq, cgpa in zip(data["IQ"], data["CGPA"])
]

df = pd.DataFrame(data)
df.to_csv("placement_data.csv", index=False)

print("✅ Dataset saved as placement_data.csv")
