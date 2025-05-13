import argparse
import signal
import sys
import numpy as np
import plotly.express as px
import pandas as pd
from utils import load_csv_np


class PairPlot:
  def __init__(self, input):
    self.data = load_csv_np(input)

  def compute(self):
    header = self.data[0]
    rows = self.data[1:]

    data_dict = {}
    for i, col_name in enumerate(header):
        column = rows[:, i]

        column = np.array([x if x != '' else np.nan for x in column], dtype=object)

        try:
            column = column.astype(float)
            data_dict[col_name] = column
        except ValueError:
            data_dict[col_name] = column
    df = pd.DataFrame(data_dict)

    numeric_cols = df.select_dtypes(include="number").columns

    if "House" in df.columns:
        color = df["House"]
    else:
        color = None

    fig = px.scatter_matrix(
        df,
        dimensions=numeric_cols,
        color=color,
        title="Pair Plot of Numeric Features",
        height=1000,
        width=1000
    )

    fig.update_traces(diagonal_visible=False)
    fig.show()



def optparse():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--input',
      '-i',
      action="store",
      dest="input",
      default="resources/dataset_train.csv",
      help="Set the input path file"
  )
  return parser.parse_args()


def signal_handler(sig, frame):
  sys.exit(0)


if __name__ == '__main__':
  signal.signal(signal.SIGINT, signal_handler)
  option = optparse()
  PairPlot(option.input).compute()
