import argparse
from math import sqrt
import signal
import numpy as np
from utils import *

class Describe:
  def __init__(self, input):
    
    self.data = load_csv_np(input)
  
  def print_describe(self, name, count, mean, std, min, first, median, third, max):
    print(name)
    print('Count : ' + str(count))
    print('Mean : ' + str(mean))
    print('Std : ' + str(std))
    print('Min : ' + str(min))
    print('25% : ' + str(first))
    print('50% : ' + str(median))
    print('75% : ' + str(third))
    print('Max : ' + str(max))
    
  
  def describe(self):
    header = self.data[0]
    rows = self.data[1:]
    
    for i in range(self.data.shape[1]):
      try:
        numeric_col = np.array(rows[:, i], dtype=float)
        numeric_col = np.sort(numeric_col[~np.isnan(numeric_col)])

        n = len(numeric_col)
        first_quartile_index = 25 * n / 100
        median_index = 50 * n / 100
        third_quartile_index = 75 * n / 100
        sum = 0
        std = 0

        for x in numeric_col:
          sum += x

        mean = sum / n

        for j in range(n):
          if j < first_quartile_index:
            first_quartile = numeric_col[j]
          if j < median_index:
            median = numeric_col[j]
          if j < third_quartile_index:
            third_quartile = numeric_col[j]

          std += (mean - numeric_col[j]) ** 2

        std = np.sqrt(std/(n - 1))
        self.print_describe(header[i], n, mean, std, numeric_col[0], first_quartile, median, third_quartile, numeric_col[n - 1])
      except:
        print(f"No numerical value for column {header[i]}")
              
        
        
    else:
        print(f"{header[i]}: non-numeric")

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
  
  
  Describe(option.input).describe()