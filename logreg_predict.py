import argparse
import signal
import sys

from sklearn.metrics import accuracy_score

import pandas as pd

from utils import *

HOUSES = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']

class LogRegPred:
  def __init__(self, input, thetas, features, output):
    self.thetas = load_thetas(thetas)
    
    
    self.input = input
    self.output = output
    self.df = load_csv_df(self.input)
    self.features = load_features(features)
    
    self.X = pd.DataFrame()
    self.y = pd.DataFrame()
  
  def load_and_clean_data(self):
    self.df = self.df.dropna(subset=self.features + ["Hogwarts House"])
    self.X = self.df[self.features].values
    self.y = self.df["Hogwarts House"].values
  
  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def predict(self):
    self.load_and_clean_data()
    m, n = self.X.shape
    X_bias = np.c_[np.ones((m, 1)), self.X]
    
    probabilities = {}
    
    for house, thetas in self.thetas.items():
      z = X_bias.dot(np.array(thetas))
      probabilities[house] = self.sigmoid(z)
      
    predictions = []
    
    for i in range(m):
      house_probability = {house: probabilities[house][i] for house in self.thetas}
      predicted_house = max(house_probability, key=house_probability.get)
      predictions.append(predicted_house)
    
    print(f"Prediction with features : {self.features}")
    print(accuracy_score(self.y, predictions))
    
    df_result = pd.DataFrame({
      "Index": self.df["Index"],
      "Hogwarts House": predictions
    })
    df_result.to_csv(self.output, index=False)
      
def optparse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-in', action="store", dest="input", default="resources/dataset_train.csv", help="select the input file")
  parser.add_argument('--thetas', '-t', action="store", dest="thetas", default="resources/thetas.txt", help="select the thetas file")
  parser.add_argument('--features', '-f', action="store", dest="features", default="resources/features.txt", help="select the features file")
  parser.add_argument('--output', '-o', action="store", dest="output", default="resources/house.csv", help="select the output file")

  return parser.parse_args()


def signal_handler(sig, frame):
    sys.exit(0)

if __name__ == '__main__':
  signal.signal(signal.SIGINT, signal_handler)
  options = optparse()

  print('Start predicting with this values')
  print('     Input       : ' + str(options.input))
  print('     Features    : ' + str(options.features))
  print('     Thetas      : ' + str(options.thetas))
  print('     Output      : ' + str(options.output))

  LogRegPred(options.input, options.thetas, options.features, options.output).predict()
