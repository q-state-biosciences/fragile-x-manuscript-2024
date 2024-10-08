# -*- coding: utf-8 -*-
"""
Created on Tue Jul  22 11:39

@author: Nathaniel.Delaney-Busch
"""
# %% Imports

import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.decomposition import SparsePCA
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, GridSearchCV
from sklearn.metrics import f1_score
import pycm


# %% Variables and Settings

# Folder locations
mainDir = "SOURCE FOLDER HERE"
dataLocation = mainDir + "pat_phenotyping_3rounds_raw.csv"
featureList = mainDir + "colList_well.csv"
resultsSubfolder = "results\\"
cellIds = {"patient": ["121.01", "125.01", "129.01"]}

os.makedirs(mainDir + resultsSubfolder, exist_ok=True)

# When fitting sparse PCA, how large should outlier values be before they are removed?
# Expressed as a number of standard deviations.
SIGMA_OUTLIER = 6

# Minimal mean cross-validated F1 score for individual features to be retained
SCORE_CUTOFF = 0.5

# When optimizing enet parameters, what is the search grid?
l1_rs = [0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99]
alphas = [0.01, 0.03, 0.1, 0.3, 1, 10, 100]
TEST_TOLERANCE = 0.015  # how much performance are we willing to lose in order to favor more regularization?


# %% Load and digest data


def sft_to_xy(
    sft, xcols, ycol, median_impute=False, standardizeX=False, binarizeY=False
):
    """Turn a pandas data frame into X and Y variables for sklearn."""
    if not isinstance(sft, pd.DataFrame):
        raise TypeError("SFT should be a pandas data frame")
    if not isinstance(xcols, list):
        raise TypeError(
            "xcols should be a list of strings corresponding with the desired column names"
        )
    if not isinstance(ycol, str):
        if isinstance(ycol, list) and len(ycol) == 1:
            ycol = ycol[0]
        if not isinstance(ycol, str):
            raise TypeError("ycol should be a string.")
    if ycol in sft:
        if sum(sft[ycol].isna()) > 0:
            warnings.warn(
                "NAs detected in the outcome column, removing rows without a label."
            )
            sft = sft.loc[-sft[ycol].isna()]
        Y = sft.loc[:, ycol]
        if len(Y.unique()) > 2:
            warnings.warn("Warning: more than two levels detected in outcome variable.")
            print(Y.unique())
        elif binarizeY:
            lb = LabelBinarizer()
            Y = lb.fit_transform(Y)
    else:
        raise ValueError(str(ycol) + " not found in SFT")
    if median_impute == True:
        sft = sft.loc[:, xcols].fillna(sft.loc[:, xcols].median())
    if set(xcols).issubset(sft.columns):
        X = sft.loc[:, xcols]
    else:
        warnings.warn(
            "Some feature colnames were not found in the SFT, using "
            "only those features that we can find. Missing Features: "
            + str([cols for cols in xcols if cols not in sft])
        )
        xcols = [cols for cols in xcols if cols in sft]
        X = sft.loc[:, xcols]
    if standardizeX == True:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    return X, Y


# %% Load data
patDFraw = pd.read_csv(dataLocation)
patDF = patDFraw.copy()
patDF.reset_index(drop=True, inplace=True)

# Load data cols
dataCols = pd.read_csv(featureList)
dataCols = list(dataCols.Var1)
dataCols = [col for col in dataCols if col != "healthySources"]
dataCols.append("highSpikerProportion")


# %% Fit new sparse PCA

print("\nNA values:")
print(patDF[dataCols].isna().sum().sort_values())
X_pat, Y_pat = sft_to_xy(
    patDF, dataCols, "diagnosis", median_impute=True, standardizeX=False
)

zScoreScaler = StandardScaler()
zScoreScaler.fit(X_pat)
X_pat_z = zScoreScaler.transform(X_pat)

# Fit the manifold with outlier-trimmed data
outlierLogical = np.abs(X_pat_z) > SIGMA_OUTLIER
print("\n\nOutliers per feature:")
print(pd.DataFrame(zip(dataCols, outlierLogical.sum(axis=0))).sort_values(1).tail(20))
print("\n\nOutliers per row:")
print(
    pd.DataFrame(
        zip(patDF["genotypePretty"], patDF["ImagingStart"], outlierLogical.sum(axis=1))
    )
    .sort_values(2)
    .tail(20)
)
for columnIdx in range(1, outlierLogical.shape[1]):
    columnLogical = outlierLogical[:, columnIdx]
    if any(columnLogical):
        X_pat.iloc[columnLogical, columnIdx] = np.nan

X_pat = X_pat.fillna(X_pat.median())

sparse_model = SparsePCA(random_state=2019, n_components=10)
sparse_model.fit(X_pat_z)
localPCACols = ["sPCA_local_" + str(comp_num) for comp_num in range(1, 11)]

# Look at components
pcaCoefs = pd.DataFrame(sparse_model.components_, columns=dataCols).transpose()
easyRead = pcaCoefs[pcaCoefs > 0.05]
pcaCoefs.to_csv(mainDir + resultsSubfolder + "pcaCoefs.csv")
easyRead.to_csv(mainDir + resultsSubfolder + "pcaCoefs_easyRead.csv")

# Compuote the sPCA scores on original data (without outlier removal)
X_pat, Y_pat = sft_to_xy(
    patDFraw, dataCols, "diagnosis", median_impute=True, standardizeX=False
)
X_pat_all_z = zScoreScaler.transform(X_pat)
componentScores_all = sparse_model.transform(X_pat_all_z)
componentScoresDataFrame = pd.DataFrame(componentScores_all, columns=localPCACols)
patDFraw = pd.concat([patDFraw, componentScoresDataFrame], axis=1)
patDF = patDFraw.copy()
patDF.reset_index(drop=True, inplace=True)
allCols = dataCols + localPCACols

# %% Data Prep
# Subtract groupwise means per round, center at grand mean of controls
totalDF = patDFraw.copy()
totalDF.to_csv(mainDir + resultsSubfolder + "dataStack.csv")

totalCTRL = totalDF.loc[totalDF.diagnosis == "CTRL"]
ctrlGroups = totalCTRL.groupby(["patientPair", "replicateRound"]).mean()
ctrlGroups = ctrlGroups.apply(lambda x: x.fillna(x.median()), axis=0)
# corrections are applied using the multi-index functionality of pandas (the
# rows of trainDF will be operated on by the appropriate patient pairs and replicate
# rounds of ctrlGroups).
trainDF = totalDF.set_index(["patientPair", "replicateRound", "ExperimentID"])
trainDF[allCols] = trainDF[allCols] - ctrlGroups[allCols] + ctrlGroups[allCols].mean()
trainDF.reset_index(inplace=True)
trainDF.to_csv(mainDir + resultsSubfolder + "centeredDataStack.csv")

# Visualize some features before and after alignment. Note that FXS cases from
# each patient pair are aligned using its corresponding control.
sns.set()
my_pal = {"FXS": "r", "CTRL": "b"}
sns.relplot(
    x="sPCA_local_2",
    y="sPCA_local_5",
    style="patientPair",
    hue="diagnosis",
    palette=my_pal,
    row="replicateRound",
    data=patDF,
)
plt.xlim(-30, 30)
plt.ylim(-20, 20)
plt.gcf().set_size_inches(10, 7)
plt.show()

sns.set()
my_pal = {"FXS": "r", "CTRL": "b"}
sns.relplot(
    x="sPCA_local_2",
    y="sPCA_local_5",
    style="patientPair",
    hue="diagnosis",
    palette=my_pal,
    row="replicateRound",
    data=trainDF,
)
plt.xlim(-30, 30)
plt.ylim(-20, 20)
plt.gcf().set_size_inches(10, 7)
plt.show()


# %% Eliminate extremely colinear features
def eliminateByCorrelation(sft, xcols, corrThreshold=0.90, redundantFeatureCutoff=2):
    """Given a feature table (or a path to one) and a starting set of column names,
    this function iteratively calculates the correlation matrix and eliminates
    features that are highly redundant. Redundant features are determined by counting
    the number of other features it is correlated with beyond some threshold
    (the corrThreshold). If enough features are highly correlated with it (beyond
    the redundantFeatureCutoff) it is elected as an elimination candidate. After
    checking all features, those with the MOST other highly correlated features
    are maintained as elimination candidates (e.g. a set of 10 features all highly
    correlated with one another). Finally, the feature with the highest average
    absolute correlation with the others is removed. The iterations cease when
    there are no more eligible features to trim."""

    # Parse xcols
    if isinstance(xcols, dict):
        # Don't accidentally try to pass non-numeric columns when not using explicit column specification.
        df = sft.select_dtypes(include=["float64", "int64", "float32", "int32"])
        if "startswith" in xcols.keys():
            xcols = [col for col in df if col.startswith(xcols["startswith"])]
        elif "endswith" in xcols.keys():
            xcols = [col for col in df if col.endswith(xcols["endswith"])]

    # Parse thresholds
    if type(corrThreshold) != float:
        corrThreshold = float(corrThreshold)
    assert (
        corrThreshold < 1 and corrThreshold > 0
    ), "corrThreshold must be between 0 and 1 (we use absolute values)"

    if type(redundantFeatureCutoff) != int:
        redundantFeatureCutoff = int(corrThreshold)
    assert redundantFeatureCutoff > 0 and corrThreshold < len(
        xcols
    ), "redundantFeatureCutoff must be between 1 and the number of features in your xcols"

    # Make copy of xcols values to prevent mutable list from being changed in the parent namespace
    localxCols = xcols[:]

    # Iteration loop
    continueEliminatingFeatures = True
    iterations = 0
    while continueEliminatingFeatures:
        correlationMatrix = sft[localxCols].corr()
        redundancyCount = correlationMatrix[correlationMatrix > corrThreshold].count()
        if max(redundancyCount) > redundantFeatureCutoff:
            highestRedundancyFeatures = redundancyCount == max(redundancyCount)
            meanCorrelations = (
                correlationMatrix[highestRedundancyFeatures].abs().mean(axis=1)
            )
            eliminationCandidates = meanCorrelations == max(meanCorrelations)
            eliminationNames = correlationMatrix[highestRedundancyFeatures][
                eliminationCandidates
            ].index.tolist()
            featureToEliminate = random.choice(eliminationNames)
            localxCols.remove(featureToEliminate)
            iterations += 1
            if iterations % 50 == 0:
                print(
                    "Eliminated "
                    + str(iterations)
                    + " features. "
                    + str(len(localxCols))
                    + " remaining to check."
                )
        else:
            continueEliminatingFeatures = False
            print(
                "All done. Eliminated "
                + str(iterations)
                + " features. "
                + str(len(localxCols))
                + " returned."
            )
    return localxCols


# Reduce feature set by correlation
print("Reduce features by correlation")
dataColsReduced = eliminateByCorrelation(trainDF, dataCols)
dataColsML = dataColsReduced + localPCACols


# %% Find optimal enet params
print("Findind Optimal Parameters for Elastic Net")
X_train, Y_train = sft_to_xy(
    trainDF, dataColsReduced, "diagnosis", median_impute=True, standardizeX=False
)
scaleToTrain = StandardScaler()
scaleToTrain.fit(X_train)
X_train_z = scaleToTrain.transform(X_train)

parameters = {
    "alpha": alphas,
    "l1_ratio": l1_rs,
}
cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    SGDClassifier(
        class_weight=None, loss="modified_huber", penalty="elasticnet", max_iter=2000
    ),
    parameters,
    scoring="roc_auc",
    cv=cv,
    return_train_score=True,
)
grid_search.fit(X_train_z, Y_train)
gridScores = pd.DataFrame(grid_search.cv_results_)
gridScores["overfitting"] = gridScores.mean_train_score - gridScores.mean_test_score
maxTestScoresPerL1 = (
    gridScores[["param_l1_ratio", "mean_test_score"]]
    .groupby("param_l1_ratio")
    .max()
    .reset_index()
)
maxTestScoresPerL1["mean_test_score"] = (
    maxTestScoresPerL1["mean_test_score"] - maxTestScoresPerL1["mean_test_score"].max()
)
maxL1Value = maxTestScoresPerL1.loc[
    maxTestScoresPerL1.mean_test_score > (0 - TEST_TOLERANCE), "param_l1_ratio"
][-1:]

gridScoresWithL1 = gridScores[gridScores.param_l1_ratio.values == maxL1Value.values]
maxAlphaValue = gridScoresWithL1.loc[
    gridScoresWithL1.mean_test_score
    > (gridScoresWithL1.mean_test_score.max() - TEST_TOLERANCE * 2),
    "param_alpha",
][-1:]

best_params = {"alpha": maxAlphaValue.values, "l1_ratio": maxL1Value.values}

# %% Elastic net importance
print("Looping Over Patient data sets")
patientPairs = trainDF.patientPair.unique()
testKappa = np.zeros(len(patientPairs))
rankingsPerFold = pd.DataFrame(columns=patientPairs, index=dataColsReduced)
rankingsPerFold.reset_index(inplace=True)
for pairIdx, holdoutPair in enumerate(patientPairs):
    X_train, Y_train = sft_to_xy(
        trainDF.loc[trainDF.patientPair != holdoutPair],
        dataColsReduced,
        "diagnosis",
        median_impute=True,
        standardizeX=False,
    )
    X_test, Y_test = sft_to_xy(
        trainDF.loc[trainDF.patientPair == holdoutPair],
        dataColsReduced,
        "diagnosis",
        median_impute=True,
        standardizeX=False,
    )
    scaleToTrain = StandardScaler()
    scaleToTrain.fit(X_train)
    X_train_z = scaleToTrain.transform(X_train)
    X_test_z = scaleToTrain.transform(X_test)

    enetModel = SGDClassifier(
        class_weight="balanced",
        loss="modified_huber",
        penalty="elasticnet",
        alpha=best_params["alpha"],
        l1_ratio=best_params["l1_ratio"],
        n_jobs=2,
        max_iter=3000,
    )
    enetModel.fit(X_train_z, Y_train)

    # get rankings
    rankings = pd.DataFrame(
        zip(dataColsReduced, enetModel.coef_.ravel()), columns=["Features", "Coefs"]
    )
    rankings["Importance"] = np.abs(rankings.Coefs)
    rankingsPerFold[[holdoutPair]] = rankings[["Importance"]]

    cm = pycm.ConfusionMatrix(Y_test.ravel(), enetModel.predict(X_test_z), digit=5)
    testKappa[pairIdx] = cm.Kappa


testKappa = np.where(testKappa < 0, 0, testKappa)
foldWeights = testKappa / sum(testKappa)
weightedImportances = pd.DataFrame(
    zip(
        dataColsReduced,
        rankingsPerFold[patientPairs].apply(np.average, weights=foldWeights, axis=1),
    ),
    columns=["Features", "weightedImportance"],
)
weightedImportances.sort_values(by="weightedImportance", ascending=False, inplace=True)
weightedImportances.reset_index(drop=True, inplace=True)
weightedImportances.to_csv(mainDir + resultsSubfolder + "patient_Importances.csv")

# Plot the feature importances of the elastic net
fig, ax = plt.subplots(figsize=(10, 10))
sns.barplot(
    y="Features",
    x="weightedImportance",
    data=weightedImportances.loc[weightedImportances.weightedImportance > 0],
)
plt.title("Feature importances")
plt.tight_layout()
