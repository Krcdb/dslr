import argparse
import signal
import sys
import plotly.graph_objects as go

import numpy as np

from utils import load_csv_np


class Histogram:
  def __init__(self, input):
    self.data = load_csv_np(input)
  
  def compute(self):
    header = self.data[0]
    rows = self.data[1:]
    houses = rows[:, 1]
    
    print(houses)
    
    for i in range (self.data.shape[1]):
      try:
        col = np.array(rows[:, i], dtype=float)
        col = col[~np.isnan(col)]
        
        h = {
          "Gryffindor": np.array([]),
          "Slytherin": np.array([]),
          "Ravenclaw": np.array([]),
          "Hufflepuff": np.array([])
        }
        
        for j in range(len(col)):
          h[houses[j]] = np.append(h[houses[j]], col[j])
          
        fig = go.Figure()

        for house, values in h.items():
            fig.add_trace(go.Histogram(
                x=values,
                name=house,
                opacity=0.7
            ))

        fig.update_layout(
            title=f"Score Distribution for {header[i]}",
            xaxis_title="Score",
            yaxis_title="Count",
            barmode="overlay"
        )

        fig.show()
        
      except Exception as e:
        print(e)
        print(f"No numerical value for column {header[i]}")


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
  
  
  Histogram(option.input).compute()