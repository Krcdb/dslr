import argparse
import signal
import sys
import numpy as np
import pandas as pd
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import load_csv_np


class ScatterPlotGrid:
  def __init__(self, input_path):
    self.data = load_csv_np(input_path)

  def compute(self):
    header = self.data[0]
    rows = self.data[1:]
    df = pd.DataFrame(rows, columns=header)

    for col in df.columns:
      df[col] = pd.to_numeric(df[col], errors='ignore')

    house_col = next((col for col in df.columns if "house" in col.lower()), None)
    if house_col is None:
        print("Error: 'Hogwarts House' column not found.")
        return

    numeric_cols = df.select_dtypes(include='number').columns
    df = df[[house_col] + list(numeric_cols)].dropna()

    feature_pairs = list(itertools.combinations(numeric_cols, 2))
    num_plots = len(feature_pairs)

    cols = 3
    rows = (num_plots + cols - 1) // cols

    fig = make_subplots(
      rows=rows,
      cols=cols,
      subplot_titles=[f"{x} vs {y}" for x, y in feature_pairs],
      horizontal_spacing=0.04,
      vertical_spacing=0.02
    )


    house_colors = {
        'Ravenclaw': 'blue',
        'Slytherin': 'green',
        'Gryffindor': 'red',
        'Hufflepuff': 'orange'
    }

    for i, (x_feat, y_feat) in enumerate(feature_pairs):
      r = i // cols + 1
      c = i % cols + 1

      for house in df[house_col].unique():
        house_df = df[df[house_col] == house]
        fig.add_trace(
          go.Scatter(
            x=house_df[x_feat],
            y=house_df[y_feat],
            mode='markers',
            name=house if i == 0 else None,
            marker=dict(color=house_colors.get(house, 'gray'), size=5),
            legendgroup=house,
            showlegend=(i == 0),
            opacity=0.5
          ),
          row=r,
          col=c,
        )

    fig.update_layout(
        height=300 * rows,
        width=1000,
        title_text="Scatter Plot Grid of Numeric Features by House",
        showlegend=True
    )
    fig.show()


def optparse():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input', '-i',
      default="resources/dataset_train.csv",
      help="Set the input path file"
  )
  return parser.parse_args()


def signal_handler(sig, frame):
  sys.exit(0)


if __name__ == '__main__':
  signal.signal(signal.SIGINT, signal_handler)
  args = optparse()
  ScatterPlotGrid(args.input).compute()
