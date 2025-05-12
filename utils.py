import csv
import os
import sys
import pandas as pd
import numpy as np

def load_csv_np(input):
  dataset = list()
  with open(input) as csvfile:
    reader = csv.reader(csvfile)
    try:
      for _ in reader:
        row = list()
        for value in _:
          try:
            value = float(value)
          except:
            if not value:
              value = np.nan
          row.append(value)
        dataset.append(row)
    except csv.Error as e:
      print(f'file {input}, line {reader.line_num}: {e}')
  return np.array(dataset, dtype=object)

def load_csv_df(input):
  if not os.path.isfile(input):
    print(f"Error: File '{input}' does not exist.")
    sys.exit(1)
  
  try:
    df = pd.read_csv(input)
  except Exception as e:
    print(f"Error reading file '{input}': {e}")
    sys.exit(1)
    
  return df
