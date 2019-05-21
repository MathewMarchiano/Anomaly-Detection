from AnomalyDetection.DataManagement import DatasetHandler
from AnomalyDetection.DataManagement import ThresholdManager
from AnomalyDetection.DataManagement import DataProcessor
from AnomalyDetection.Splitter import Splitter
from AnomalyDetection.Trainer import Trainer
import numpy as np

#Letter Recognition
cb = [[1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0,
   1, 0, 0], [1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
   0, 1, 1, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1,
   1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0], [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
   0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 0, 0, 0, 1,
   0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0], [1, 0, 0,
   0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1,
  1], [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0,
  0, 1, 0, 1, 1], [1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0,
  0, 1, 0, 0, 0, 1, 0, 1, 0], [0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1,
  0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0], [0, 1, 1, 0, 0, 1, 1, 0, 0,
  1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 1,
  0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0], [1,
  0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
   1, 0], [0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0,
   0, 0, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1,
   1, 1, 0, 1, 0, 0, 1, 0, 0, 1], [0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1,
   0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 0, 1, 0,
   1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1], [0, 1, 0, 1,
   1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0,
  1], [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
  0, 0, 0, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1,
  0, 0, 0, 1, 0, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
  0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1], [0, 1, 1, 1, 1, 1, 0, 1, 0,
  0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0], [1, 1, 1, 1, 0,
  1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], [1,
  0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1,
   1, 0], [1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1,
   1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0,
   0, 1, 1, 1, 1, 1, 0, 1, 1, 0], [0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0,
   1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1]]

cb2 = [[0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
   1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1], [1, 0, 1, 1, 0, 0,
   0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0,
  1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1,
   0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1,
   1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
  0], [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1,
  0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0,
  0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0,
   1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 1, 1, 1, 0,
   1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0,
  0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0,
  1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,
   0], [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1,
   0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0], [0, 0, 1, 0,
   0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
  0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1], [1, 0, 1, 1, 1, 0, 1, 0, 0,
  1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
   0, 1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0,
   1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1,
  0, 0], [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
  1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1], [0, 1, 0,
  1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1,
   1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 0, 1, 0, 0, 1,
   1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0,
  1, 1, 1, 0, 0, 1, 0, 1], [0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1,
  0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0,
   0, 0], [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0,
   0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1], [0, 1, 0,
   1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0,
  0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 1, 0, 1, 1, 0,
  0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0,
   0, 0, 0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0,
   0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
  0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1,
  1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0], [1, 0,
  1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
   1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0], [1, 1, 1, 0, 1, 0, 1, 1,
   0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
  1, 0, 0, 1, 0, 0, 1, 1, 0], [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0,
  1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1,
   0, 1, 0], [0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1,
   1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1], [1, 1,
   0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
  1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]]

cb3 = [[0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0,
   1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0,
  0, 0, 1, 1, 1, 0, 0], [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1,
  0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0,
   0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 1, 0, 1, 1, 0,
   1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0,
  1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1,
  1], [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0,
  1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1,
   0, 0, 0, 1, 0, 0, 1, 1], [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
   1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
  0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], [1, 0, 1, 1, 1, 0,
  0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0,
   1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
  1], [0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1,
  1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
   0, 1, 0, 1, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1,
   1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1,
  0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1], [1, 0, 1, 1, 0, 1,
  0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0,
   1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
  1], [1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0,
  0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0,
   1, 1, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
   1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0], [0, 0, 0, 1, 0, 1,
  0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1,
   1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
  1], [0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1,
  1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1,
   1, 0, 1, 0, 0, 1, 1, 1], [1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
   0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1,
  0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0], [1, 0, 1, 0, 1, 1,
  0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
   0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
  1], [0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1,
  0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
   0, 1, 0, 1, 1, 0, 1, 0], [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1,
   1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0,
  1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0], [1, 1, 0, 1, 1, 0,
  1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1,
   1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0,
  1], [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
  0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0,
   1, 1, 0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
   1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0,
  0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0,
  1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
   1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0,
  1], [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1,
  1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0,
   1, 1, 0, 0, 1, 1, 0, 0], [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1,
   1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,
  0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 1, 1,
  1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0,
   1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1,
  1], [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,
  0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,
   0, 0, 1, 0, 0, 1, 1, 0], [1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1,
   1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1,
  0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0]]

dataset= "https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
listOfThresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
listOfSplits = [0.10, .15, .20, .25]
listOfCBs = [cb, cb2, cb3]


def loop(listOfCBs, listOfThresholds, listOfNewSplits, dataset, numHoldouts):
     iterationCount = 1 # To make sure everything is running and to keep track of what iteration you are on.
                        # Not really necessary for proper functioning of the program.
     optimalThresholds = []
     listOfDifferences = []
     unknownAccuracies = []
     knownAccuracies = []
     # Max
     unknownMaxAccDictionary = {}
     knownMaxAccDictionary = {}
     thresholdMaxDictionary = {}
     # Min
     unknownMinAccDictionary = {}
     knownMinAccDictionay = {}
     thresholdMinDictionary = {}
     # Var
     unknownVarDictionary = {}
     knownVarDictionary = {}
     thresholdVarDictionary = {}
     # Means
     unknownMeanDictionary = {}
     knownMeanDictionary = {}
     thresholdMeanDictionary = {}

     codebookNum = 0
     splitter = Splitter()
     trainer = Trainer()
     tm = ThresholdManager()
     dp = DataProcessor()
     for codebook in listOfCBs:
         dh = DatasetHandler(codebook)
         codebookNum += 1
         for split in listOfNewSplits:
             for holdout in range(numHoldouts):
                 allData, allOriginalLabels = dh.getData(dataset, -1, 1, 7)
                 scaledData = dh.preprocessData(allData)
                 ECOCLabels, labelDictionary = dh.assignCodeword(allOriginalLabels)
                 binaryClassifiers = dh.binarizeLabels(labelDictionary)
                 listOfUnknownClasses, listOfKnownClasses, holdoutClass = \
                     splitter.assignLabel(allOriginalLabels, split, holdout)
                 knownValidationData, knownValidationLabels, singleDataSamples, singleDataSamplesLabels, knownData, \
                 knownLabels, unknownData, unknownLabels, holdoutData, holdoutLabels \
                     = splitter.splitDataAndLabels(scaledData, allOriginalLabels, listOfUnknownClasses, holdoutClass)

                 knownECOCLabels = trainer.makeTrainingLabels(binaryClassifiers, knownLabels)
                 listOfClassifiers = trainer.trainClassifiers(knownData, knownECOCLabels, 4)
                                                            #1 = SVM, 2 = DT, 3 = LDA, 4 = KNN

                 # Getting predictions on all relevant data
                 # knownValidationData refers to the 20% split from the known data used for training
                 # validationData refers to the data that is split from the data before training accutally
                 # happens using sklearn's train_test_split().
                 unknownPreds = trainer.getPredictions(unknownData, listOfClassifiers)
                 holdoutClassPreds = trainer.getPredictions(holdoutData, listOfClassifiers)
                 singleDataSamplesPreds = trainer.getPredictions(singleDataSamples, listOfClassifiers)
                 knownValidationPreds = trainer.getPredictions(knownValidationData, listOfClassifiers)

                 # Getting the shortest hamming distance that each prediction corresponds to:
                 unknownECOCPreds, unknownHDs = trainer.hammingDistanceUpdater(codebook, unknownPreds)
                 holdoutClassECOCPreds, holdoutClassHDs = trainer.hammingDistanceUpdater(codebook, holdoutClassPreds)
                 singleDataSamplesECOCPreds, singleDataSamplesHDs = trainer.hammingDistanceUpdater(codebook,
                                                                                                singleDataSamplesPreds)
                 knownValidationECOCPreds, knownValidationHDs = trainer.hammingDistanceUpdater(codebook,
                                                                                               knownValidationPreds)

                 optimalThreshold, lowestDifference, highestKnownAcc, highestUnknownAcc = \
                                                 tm.findOptimalThreshold(listOfThresholds, knownValidationHDs, unknownHDs)

                 # Getting accuracies of predictions (whether known or unknown)
                 knownHoldoutDataThresholdAcc = dp.knownThresholdTest(singleDataSamplesHDs, optimalThreshold)
                 unknownHoldoutDataThresholdAcc = dp.unknownThresholdTest(holdoutClassHDs, optimalThreshold)


                 print("Codebook:", codebookNum, "split:", split, "iteration:", iterationCount)
                 iterationCount += 1

                 optimalThresholds.append(optimalThreshold)
                 listOfDifferences.append(lowestDifference)
                 unknownAccuracies.append(unknownHoldoutDataThresholdAcc)
                 knownAccuracies.append(knownHoldoutDataThresholdAcc)

                 #Graphing to see how threshold is performing:
                 dp.graphKnownUnknownHDs(singleDataSamplesHDs, holdoutClassHDs, optimalThreshold, codebookNum,
                                         split, knownHoldoutDataThresholdAcc, unknownHoldoutDataThresholdAcc,
                                         12, holdoutClass, allData, unknownData, knownData, codebook, singleDataSamples)

             print("Mean of Optimal Thresholds:", np.mean(optimalThresholds))
             print("Max Threshold:", max(optimalThresholds))
             print("Min Threshold:", min(optimalThresholds))
             print("Thresholds Variance:", np.var(optimalThresholds), "\n")
             thresholdMaxDictionary[split] = max(optimalThresholds)
             thresholdMinDictionary[split] = min(optimalThresholds)
             thresholdVarDictionary[split] = np.var((optimalThresholds))
             thresholdMeanDictionary[split] = np.mean(optimalThresholds)

             print("Mean of Known Accuracies:", np.mean(knownAccuracies))
             print("Max Known Accuracy:", max(knownAccuracies))
             print("Min Known Accuracy:", min(knownAccuracies))
             print("Known Accuracies Variance:", np.var(knownAccuracies), "\n")
             knownMaxAccDictionary[split] = max(knownAccuracies)
             knownMinAccDictionay[split] = min(knownAccuracies)
             knownVarDictionary[split] = np.var(knownAccuracies)
             knownMeanDictionary[split] = np.mean(knownAccuracies)

             print("Mean of New Accuracies:", np.mean(unknownAccuracies))
             print("Max Unknown Accuracy:", max(unknownAccuracies))
             print("Min Unknown Accuracy:", min(unknownAccuracies))
             print("Unknown Accuracies Variance:", np.var(unknownAccuracies), "\n")
             unknownMaxAccDictionary[split] = max(unknownAccuracies)
             unknownMinAccDictionary[split] = min(unknownAccuracies)
             unknownVarDictionary[split] = np.var(unknownAccuracies)
             unknownMeanDictionary[split] = np.mean((unknownAccuracies))

             optimalThresholds = []
             unknownAccuracies = []
             knownAccuracies = []
             iterationCount = 1

         dp.accuraciesPlot(knownMinAccDictionay, knownMaxAccDictionary, unknownMinAccDictionary,
                           unknownMaxAccDictionary,knownMeanDictionary, unknownMeanDictionary,
                           codebook, knownData, allData, unknownData, singleDataSamples)

         # Max
         unknownMaxAccDictionary = {}
         knownMaxAccDictionary = {}
         thresholdMaxDictionary = {}
         # Min
         unknownMinAccDictionary = {}
         knownMinAccDictionay = {}
         thresholdMinDictionary = {}
         # Var
         unknownVarDictionary = {}
         knownVarDictionary = {}
         thresholdVarDictionary = {}
         # Means
         unknownMeanDictionary = {}
         knownMeanDictionary = {}
         thresholdMeanDictionary = {}


#loop(listOfCBs, listOfThresholds, listOfSplits, dataset, 28)


