import argparse
import json
import signal
import sys
import numpy as np
import pandas as pd

from utils import *


HOUSES = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']

class LogRegTrain:
  def __init__(self, input, feature_input, output, alpha, iteration):
    self.input = input
    self.output = output
    self.alpha = alpha
    self.iteration = iteration
    
    self.X = pd.DataFrame()
    self.y = pd.DataFrame()
    
    self.mean = 0
    self.std = 0
    
    self.thetas = {}
    self.features = load_features(feature_input)
  
  def sigmoid(self, z):
    return 1 / (1 + np.exp(-z))
  
  def house_to_binary(self, house):
    return np.array([1 if label == house else 0 for label in self.y])
  
  def load_and_clean_data(self):
    df = load_csv_df(self.input)
    df = df.dropna(subset=self.features + ["Hogwarts House"])
    self.X = df[self.features].values
    self.y = df["Hogwarts House"].values
    
    self.mean = self.X.mean(axis=0)
    self.std = self.X.std(axis=0)
    
    self.X = (self.X - self.mean) / self.std
    
  def gradient_descent(self, X_bias, y, theta):
    m = len(self.y)
    
    for _ in range(self.iteration):
      z = X_bias.dot(theta)
      y_hat = self.sigmoid(z)
      
      gradient = X_bias.T.dot(y_hat - y) / m
      theta -= self.alpha * gradient
    
    return theta
  
  def unnormalize_theta(self, theta):
    theta_0 = theta[0]
    theta_rest = np.array(theta[1:])
    
    theta_new = theta_rest / self.std
    theta_0_new = theta_0 - np.sum(theta_rest * self.mean / self.std)
    
    return np.concatenate([[theta_0_new], theta_new])
  
  def save_thetas(self):
    thetas_unnorm = {house: self.unnormalize_theta(np.array(theta)).tolist()
                     for house, theta in self.thetas.items()}
    with open(self.output, 'w') as f:
      json.dump(thetas_unnorm, f, indent=2)
      
  def train(self):
    self.load_and_clean_data()
    m, n = self.X.shape
    X_bias = np.c_[np.ones((m, 1)), self.X]
    for house in HOUSES:
      y_binary = self.house_to_binary(house)
      theta = np.zeros(n + 1)
      theta = self.gradient_descent(X_bias, y_binary, theta)
      self.thetas[house] = theta.tolist()
    self.save_thetas()

def optparse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-in', action="store", dest="input", default="resources/dataset_train.csv", help="select the input file")
  parser.add_argument('--features', '-f', action="store", dest="features", default="resources/features.txt", help="select the features file")
  parser.add_argument('--output', '-o', action="store", dest="output", default="resources/thetas.txt", help="select the output file")
  parser.add_argument('--alpha', '-a', action="store", dest="alpha", default=0.01, type=float, help="set the learing rate")
  parser.add_argument('--iteration', '-it', action="store", dest="iteration", default=10000, type=int, help="set the max number of iterations")

  return parser.parse_args()


def signal_handler(sig, frame):
    sys.exit(0)

if __name__ == '__main__':
  signal.signal(signal.SIGINT, signal_handler)
  options = optparse()

  if (options.alpha > 1 or options.alpha < 0.00000001):
    options.alpha = 0.01
    
  print('Start training with this values')
  print('     Input       : ' + str(options.input))
  print('     Features    : ' + str(options.features))
  print('     Output      : ' + str(options.output))
  print('     Alpha       : ' + str(options.alpha))
  print('     Iterations  : ' + str(options.iteration))

  LogRegTrain(options.input, options.features, options.output, options.alpha, options.iteration).train()
