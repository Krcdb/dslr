import argparse
import signal
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import load_csv_np


class Scatter:
  def __init__(self, input):
    self.data = load_csv_np(input)
  
  def compute(self):
    header = self.data[0]
    rows = self.data[1:]
    houses = rows[:, 1]
    
    print(houses)
    num_columns = self.data.shape[1]
    numeric_indices = []
    numeric_names = []

    for i in range(num_columns):
      try:
          np.array(rows[:, i], dtype=float)
          numeric_indices.append(i)
          numeric_names.append(header[i])
      except ValueError:
          continue

    n = len(numeric_indices)
    cols = 2
    rows_count = (n + cols - 1) // cols

    fig = make_subplots(rows=rows_count, cols=cols, subplot_titles=numeric_names)

    for idx, col_index in enumerate(numeric_indices):
      try:
        col = np.array(rows[:, col_index], dtype=float)
        col = col[~np.isnan(col)]
        r = idx // cols + 1
        c = idx % cols + 1

        fig.add_trace(
            go.Scatter(
                y=col,
                mode='markers',
                name=header[col_index],
                marker=dict(size=6),
                showlegend=False
            ),
            row=r,
            col=c
        )
      except Exception as e:
        print(f"Error plotting column {header[col_index]}: {e}")

    fig.update_layout(
        title_text="Scatter Plots of Numeric Features",
        height=300 * rows_count,
        width=900,
        showlegend=False
    )
    fig.show()

def optparse():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--input',
    '-i',
    action="store",
    dest="input",
    default="resources/dataset_train.csv",
    help="set the input path file"
  )
  
  return parser.parse_args()

def signal_handler(sig, frame):
  sys.exit(0)

if __name__ == '__main__':
  signal.signal(signal.SIGINT, signal_handler)
  option = optparse()
  
  
  Scatter(option.input).compute()