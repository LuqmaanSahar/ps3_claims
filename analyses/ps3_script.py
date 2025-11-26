# %%
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask_ml.preprocessing import Categorizer
from glum import GeneralizedLinearRegressor, TweedieDistribution
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import auc
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from ps3.data import create_sample_split, load_transform

# %%
# load data
df = load_transform()

# %%
# Train benchmark tweedie model. This is entirely based on the glum tutorial.
weight = df["Exposure"].values
df["PurePremium"] = df["ClaimAmountCut"] / df["Exposure"]
y = df["PurePremium"]
# TODO: Why do you think, we divide by exposure here to arrive at our outcome variable?

# this is to compute the 'claim frequency per year' rather than just the raw number of claims


# TODO: use your create_sample_split function here (done)
df = create_sample_split(df, "IDpol")
train = np.where(df["sample"] == "train")
test = np.where(df["sample"] == "test")
df_train = df.iloc[train].copy()
df_test = df.iloc[test].copy()

# define the categorical variables to pass through a categoriser
categoricals = ["VehBrand", "VehGas", "Region", "Area", "DrivAge", "VehAge", "VehPower"]

predictors = categoricals + ["BonusMalus", "Density"]
glm_categorizer = Categorizer(columns=categoricals)

# extract the labels and features for train and test datasets
X_train_t = glm_categorizer.fit_transform(df[predictors].iloc[train])
X_test_t = glm_categorizer.transform(df[predictors].iloc[test])
y_train_t, y_test_t = y.iloc[train], y.iloc[test]
w_train_t, w_test_t = weight[train], weight[test]

# define the distribution and fit a GLM on the training dataset
# the parameter 1.5 for the Tweedie distribution was arbitrarily chosen
# we could include this as a parameter in a grid search to determine the best
# performing set of model parameters 
TweedieDist = TweedieDistribution(1.5)
t_glm1 = GeneralizedLinearRegressor(family=TweedieDist, l1_ratio=1, fit_intercept=True)
t_glm1.fit(X_train_t, y_train_t, sample_weight=w_train_t)

# export the model results as a dataframe
pd.DataFrame(
    {"coefficient": np.concatenate(([t_glm1.intercept_], t_glm1.coef_))},
    index=["intercept"] + t_glm1.feature_names_,
).T

# make predictions on the both datasets using our fitted model
df_test["pp_t_glm1"] = t_glm1.predict(X_test_t)
df_train["pp_t_glm1"] = t_glm1.predict(X_train_t)

# report the prediction error (deviance) in the training dataset
print(
    "training loss t_glm1:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm1"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

# report the prediction error (deviance) in the testing dataset
print(
    "testing loss t_glm1:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm1"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# total predicted vs. true claim amount on the test set
print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * t_glm1.predict(X_test_t)),
    )
)
# %%
# TODO: Let's add splines for BonusMalus and Density and use a Pipeline.
# Steps: 
# 1. Define a Pipeline which chains a StandardScaler and SplineTransformer. 
#    Choose knots="quantile" for the SplineTransformer and make sure, we 
#    are only including one intercept in the final GLM. 
# 2. Put the transforms together into a ColumnTransformer. Here we use OneHotEncoder for the categoricals.
# 3. Chain the transforms together with the GLM in a Pipeline.

# create a pipeline for numerical variables
num_pipeline = make_pipeline(
    StandardScaler(),
    SplineTransformer(n_knots=5, degree=3, include_bias=False, knots="quantile")
    ) # send numeric through a spline transformer and standard scaler

# Let's put together a pipeline
numeric_cols = ["BonusMalus", "Density"]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, numeric_cols), # send numeric variables through our numeric pipeline
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categoricals) # send categoricals through a onehotencoder
    ]
)
preprocessor.set_output(transform="pandas") # output as a pd.DataFrame

# a model pipeline to chain preprocessing -> model fitting
model_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("estimate", t_glm1)
])

# let's have a look at the pipeline
model_pipeline

# let's check that the transforms worked
model_pipeline[:-1].fit_transform(df_train)

model_pipeline.fit(df_train, y_train_t, estimate__sample_weight=w_train_t)

# save the estimates
pd.DataFrame(
    {
        "coefficient": np.concatenate(
            ([model_pipeline[-1].intercept_], model_pipeline[-1].coef_)
        )
    },
    index=["intercept"] + model_pipeline[-1].feature_names_,
).T

# make predictions using the fitted model
df_test["pp_t_glm2"] = model_pipeline.predict(df_test)
df_train["pp_t_glm2"] = model_pipeline.predict(df_train)

# print deviance for model evaluation
print(
    "training loss t_glm2:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_glm2"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_glm2:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_glm2"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_glm2"]),
    )
)

# %%
# TODO: Let's use a GBM instead as an estimator.
# Steps
# 1: Define the modelling pipeline. Tip: This can simply be a LGBMRegressor based on X_train_t from before.
# 2. Make sure we are choosing the correct objective for our estimator.


# don't include preprocessing in this pipeline
model_pipeline_lgbm = Pipeline([
    ("lgbm", LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5
    ))
])

model_pipeline_lgbm.fit(X_train_t, y_train_t, lgbm__sample_weight=w_train_t)
df_test["pp_t_lgbm"] = model_pipeline_lgbm.predict(X_test_t)
df_train["pp_t_lgbm"] = model_pipeline_lgbm.predict(X_train_t)
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

# %%
# TODO: Let's tune the LGBM to reduce overfitting.
# Steps:
# 1. Define a `GridSearchCV` object with our lgbm pipeline/estimator. Tip: Parameters for a specific step of the pipeline
# can be passed by <step_name>__param. 

# Note: Typically we tune many more parameters and larger grids,
# but to save compute time here, we focus on getting the learning rate
# and the number of estimators somewhat aligned -> tune learning_rate and n_estimators

# define the grid of parameters to search upon
param_grid = {
    "lgbm__learning_rate": [0.1, 0.05, 0.01],
    "lgbm__n_estimators": [100, 300, 500]
}

# conduct the grid search using the earlier defined model pipeline
cv = GridSearchCV(
    estimator=model_pipeline_lgbm,
    param_grid=param_grid,
    cv=3,
    scoring=None,          # LightGBM uses built-in loss internally
    n_jobs=-1
)
cv.fit(X_train_t, y_train_t, lgbm__sample_weight=w_train_t)

# make predictions using the best scoring model
df_test["pp_t_lgbm"] = cv.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm"] = cv.best_estimator_.predict(X_train_t)

# evaluate
print(
    "training loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm"]),
    )
)
# %%
# Let's compare the sorting of the pure premium predictions


# Source: https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


fig, ax = plt.subplots(figsize=(8, 8))

for label, y_pred in [
    ("LGBM", df_test["pp_t_lgbm"]),
    ("GLM Benchmark", df_test["pp_t_glm1"]),
    ("GLM Splines", df_test["pp_t_glm2"]),
]:
    ordered_samples, cum_claims = lorenz_curve(
        df_test["PurePremium"], y_pred, df_test["Exposure"]
    )
    gini = 1 - 2 * auc(ordered_samples, cum_claims)
    label += f" (Gini index: {gini: .3f})"
    ax.plot(ordered_samples, cum_claims, linestyle="-", label=label)

# Oracle model: y_pred == y_test
ordered_samples, cum_claims = lorenz_curve(
    df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
)
gini = 1 - 2 * auc(ordered_samples, cum_claims)
label = f"Oracle (Gini index: {gini: .3f})"
ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

# Random baseline
ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
ax.set(
    title="Lorenz Curves",
    xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
    ylabel="Fraction of total claim amount",
)
ax.legend(loc="upper left")
plt.plot()

# %%

#########################
##### PROBLEM SET 3 #####
#########################

# %%

# EXERCISE 1: IMPOSING MONOTONICITY CONSTRAINTS

# We expect the feature 'BonusMalus' to be monotonically increasing
# However, complex predictive models (GBM) are free to learn non-monotonic patterns and interactions
# To be consistent with our prior knowledge, we impose a monotonicity constraint on this feature.

# create a folder to save outputs
plots_dir = os.path.join(os.getcwd(), "outputs")
os.makedirs(plots_dir, exist_ok=True)

# Weighted average claim amount per BonusMalus bin
# If the data is already monotonically increasing then we have no issue
# If it is not, then interactions in the model can lead to unexpected behaviour
avg_claims = (
    df_train
    .groupby("BonusMalus")
    .apply(lambda g: np.sum(g["ClaimAmountCut"] * g["Exposure"]) / np.sum(g["Exposure"]))
)

avg_claims.plot(kind="line", marker="o")
plt.xlabel("BonusMalus")
plt.ylabel("Weighted average claim amount")
plt.title("Empirical claim cost vs BonusMalus")
plt.grid(True)
file_path = os.path.join(plots_dir, "BonusMalus.png")
plt.savefig(file_path, dpi=300, bbox_inches="tight")
plt.show()

# We observe that there are oscillations. The plot is not monotonically increasing
# We need to add a monotonicity constraint to the model.
# %%

# To add the monotonicity constraints we pass a list to LGBM to indicate features
# The list must be ordered in the same way that the model sees the features
# +1 -> monotonically increasing
# -1 -> monotonically decreasing
# 0 -> no monotonicity constraint

# Retrieve the lost of all features in the dataset
feature_names = X_train_t.columns.tolist()
feature_names

# Find the index position of BonusMalus
bonusmalus_index = feature_names.index("BonusMalus")
bonusmalus_index

# Create a list of zeroes, with a 1 at the position of BonusMalus
monotonic_constraints = [0] * len(feature_names)
monotonic_constraints[bonusmalus_index] = 1

# Pass the list of monotonicity constraints into the model pipeline
pipeline_lgbm_constrained = Pipeline([
    ("lgbm_constrained", LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.5,
        monotone_constraints=monotonic_constraints
    ))
])

# Now we continue as normal

# define parameter grid for hyperparameter tuning
param_grid = {
    "lgbm_constrained__learning_rate": [0.1, 0.05, 0.01],
    "lgbm_constrained__n_estimators": [100, 300, 500]
}

# conduct the grid search
cv_constrained = GridSearchCV(
    estimator=pipeline_lgbm_constrained,
    param_grid=param_grid,
    cv=3,
    scoring=None,          # LightGBM uses built-in loss internally
    n_jobs=-1
)
cv_constrained.fit(X_train_t, y_train_t, lgbm_constrained__sample_weight=w_train_t)

# make predictions using the best scoring model
df_test["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_test_t)
df_train["pp_t_lgbm_constrained"] = cv_constrained.best_estimator_.predict(X_train_t)

# evaluate
print(
    "training loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(y_train_t, df_train["pp_t_lgbm_constrained"], sample_weight=w_train_t)
        / np.sum(w_train_t)
    )
)

print(
    "testing loss t_lgbm_constrained:  {}".format(
        TweedieDist.deviance(y_test_t, df_test["pp_t_lgbm_constrained"], sample_weight=w_test_t)
        / np.sum(w_test_t)
    )
)

print(
    "Total claim amount on test set, observed = {}, predicted = {}".format(
        df["ClaimAmountCut"].values[test].sum(),
        np.sum(df["Exposure"].values[test] * df_test["pp_t_lgbm_constrained"]),
    )
)

# %%

# EXERCISE 2: LEARNING CURVES

# For an iterative model, we can plot its progress over iterations using learning curves
# We will re-use the best model from the cross validation in the last exercise

# Retrieve the best estimator from the cross validation
best = cv_constrained.best_estimator_

# extra step to extract just the model if it is part of a Pipeline object
if isinstance(best, Pipeline):
    lgbm_est = best.named_steps.get("lgbm_constrained", None)
    if lgbm_est is None:
        raise ValueError("Pipeline found but no step named 'lgbm_constrained'. Inspect best.named_steps.")
else:
    lgbm_est = best

# Combine the train and test sets to create a master dataset to use for model evaluation
eval_set = [(X_train_t, y_train_t), (X_test_t, y_test_t)]
eval_sample_weight = [w_train_t, w_test_t]

# set verbose for informative training
# verbose controls how much LightGBM prints during training
lgbm_est.set_params(verbose=50)  # or 0 / -1 to silence

# If your estimator came from GridSearchCV you might want to set n_estimators to a value,
# or keep the one selected by CV.
# Refit the estimator with eval_set so we can capture the learning curve
lgbm_est.fit(
    X_train_t,
    y_train_t,
    sample_weight=w_train_t,
    eval_set=eval_set,
    eval_sample_weight=eval_sample_weight,
    eval_metric="tweedie",            # metric consistent with tweedie objective
)

# Retrieve the per-iteration evaluation results from LGBM
# sklearn wrapper stores results in .evals_result_
evals_result = lgbm_est.evals_result_

# Plot the learning curve(s)
fig, ax = plt.subplots(figsize=(10, 6))

# lgb.plot_metric accepts either an evals_result dict or a booster. Here we pass evals_result.
# If you want to plot all recorded metrics, omit metric=... ; specifying metric will plot that one only.
lgb.plot_metric(evals_result, metric="tweedie", ax=ax)

ax.set_title("Learning curve (train vs validation) — Tweedie metric")
ax.grid(True)

file_path = os.path.join(plots_dir, "LGBM_Learning_Curve.png")
plt.savefig(file_path, dpi=300, bbox_inches="tight")
plt.show()
# %%
