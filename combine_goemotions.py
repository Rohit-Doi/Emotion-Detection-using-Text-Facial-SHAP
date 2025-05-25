import pandas as pd
import os

# Define paths
data_dir = 'data/full_dataset'
output_file = 'data/goemotions_train.csv'

# List of input files
input_files = [
    os.path.join(data_dir, 'goemotions_1.csv'),
    os.path.join(data_dir, 'goemotions_2.csv'),
    os.path.join(data_dir, 'goemotions_3.csv')
]

# Target emotions
target_emotions = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'shame', 'surprise']

# Combine datasets
dfs = []
for file in input_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dfs.append(df)
    else:
        print(f"File {file} not found.")

if not dfs:
    raise FileNotFoundError("No GoEmotions files found.")

combined_df = pd.concat(dfs, ignore_index=True)

# Create a single 'emotion' column by selecting the first positive emotion per row
def get_emotion(row):
    for emotion in target_emotions:
        if emotion in row and row[emotion] == 1:
            return emotion
    return None

combined_df['emotion'] = combined_df.apply(get_emotion, axis=1)

# Filter rows with target emotions and select relevant columns
combined_df = combined_df[combined_df['emotion'].isin(target_emotions)][['text', 'emotion']]

# Save to CSV
os.makedirs('data', exist_ok=True)
combined_df.to_csv(output_file, index=False)
print(f"Saved combined dataset to {output_file}")