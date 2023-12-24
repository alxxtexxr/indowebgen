import fire
import os
import glob
import pandas as pd

def main(
  dir='.',
  file_pattern='regen-*[0-9]*.json',
  output_file='regen-dirty.json',
):
  dirty_files = glob.glob(os.path.join(dir, file_pattern))
  
  combined_df = pd.DataFrame()

  # Load all the regen JSON files
  for json_file in dirty_files:
      data = pd.read_json(json_file)
      combined_df = combined_df.append(data, ignore_index=True)
  print(f'Length of uncombined dirty data: {len(combined_df)}')

  # Drop the 'token_count' column
  if 'token_count' in combined_df.columns:
      combined_df = combined_df.drop('token_count', axis=1)
      
  combined_df['curation_status'].fillna(0, inplace=True)
  combined_df['curation_message'].fillna('', inplace=True)
  
  combined_df.drop_duplicates(subset=['instruction', 'output'], inplace=True)
  print(f'Length of combined dirty data: {len(combined_df)}')
  
  combined_df.to_json(os.path.join(dir, output_file), orient='records')
  print(f'Output file saved to: {os.path.join(dir, output_file)}')

if __name__ == '__main__':
  fire.Fire(main)