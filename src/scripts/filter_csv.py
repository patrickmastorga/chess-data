import pandas as pd

print('Reading .csv...')

# Read the CSV file into a DataFrame
df = pd.read_csv('datasets/random_evals.csv')

print('Filetering data...')

# Exclude all rows where the evaluation contains a "#" checkmate symbol
criteria = ~df['Evaluation'].str.contains('#')

# Filter the DataFrame based on the criteria
filtered_df = df[criteria]

# Extract only the first word of the FEN string
first_word_column1 = filtered_df['FEN'].str.split().str[0]

# Overwrite the filtered dataframe with the new first column
filtered_df = pd.DataFrame({
    'FEN': first_word_column1,
    'Evaluation' : filtered_df['Evaluation']
})

print('Saving .csv...')

# Save the filtered DataFrame back to a CSV file
filtered_df.to_csv('datasets/random_evals_filtered.csv', index=False)

print('Done!')