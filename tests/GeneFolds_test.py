import pytest
from src.PredictionModel import PredictionModel

model = PredictionModel('Test')

assert model.numFolds == 4, "model should have 4 folds be default"
