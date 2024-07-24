import pytest
from src.PredictionModel import PredictionModel

model = PredictionModel('Test')

assert model.numFolds == 4, "Model should have 4 folds be default"

assert model.hasIndependentGeneFolds(), "Model's gene folds are not independent"

assert model.geneFoldsCoverGoldStandard(), "Not all gold standard genes are in model's gene folds"

assert model.geneFoldsHaveEqualSize(), "Model's gene folds are not of roughly equal size"