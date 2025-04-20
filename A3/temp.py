import pandas as pd
from tabulate import tabulate

# Load your CSV file
df = pd.read_csv('image_caption_scores.csv')  # replace with your actual file path

# Print as a pretty table
print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))
