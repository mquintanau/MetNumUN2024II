#!/usr/bin/env python
# coding: utf-8

import logging

import xarray as xr  # xarray for data manipulation

import qnt.data as qndata     # functions for loading data
import qnt.backtester as qnbt # built-in backtester
import qnt.ta as qnta         # technical analysis library
import qnt.stats as qnstats   # statistical functions

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

np.seterr(divide="ignore")

# ------------------- #
#  NEW IMPORT: 
# ------------------- #
from sklearn.ensemble import RandomForestClassifier
# ------------------- #

##############################################################################
#               1) LOAD S&P 500 DATA (CHANGED FROM NASDAQ-100)              #
##############################################################################

# Example: pulling ~5 years of data (you can adjust 'tail' as needed)
# CORRECT:
stock_data = qndata.stocks.load_spx_data(
    tail=365 * 5, 
    dims=("time", "field", "asset"),
    assets=[
        "SPX:AAPL", "SPX:MSFT", "SPX:AMZN", "SPX:GOOGL", "SPX:GOOG",
        "SPX:JNJ", "SPX:V", "SPX:PG", "SPX:UNH", "SPX:XOM",
        "SPX:BAC", "SPX:DIS", "SPX:MA", "SPX:HD",  "SPX:CVX",
        "SPX:KO",  "SPX:PEP", "SPX:MRK", "SPX:TSLA", "SPX:WMT"
    ]
)

##############################################################################
#               2) BUILD FEATURES (WITH NEW VARIABLES)                       #
##############################################################################

def get_features(data):
    """
    Builds a set of features used for learning:
       * Trend (ROC of 60-day LWMA)
       * MACD
       * Volatility (14-day)
       * Stochastic Oscillator
       * RSI
       * Log of closing price
       * Bollinger Bands (upper, middle, lower)
       * On Balance Volume (OBV)
       * ADX (PlusDI, MinusDI, ADX)
    
    Reasoning for New Variables:
      - Bollinger Bands capture volatility & price extremes around a moving average.
      - OBV ties volume to price movements, potentially detecting accumulation or distribution.
      - ADX (with PlusDI & MinusDI) measures trend strength and direction.
    """

    close = data.sel(field="close")
    print("close shape:", close.shape)
    print("non-NaN count:", close.count().values)  # how many non-NaN entries
    print("close coords:", close.coords)
    print(close.isnull().sum(dim="time"))

    high = data.sel(field="high")
    low  = data.sel(field="low")
    vol  = data.sel(field="vol").fillna(0)

    # --- Existing Features ---
    # Trend:
    trend = qnta.roc(qnta.lwma(close, 60), 1)

    # MACD (using default parameters as example)
    macd_line, macd_signal, macd_hist = qnta.macd(close)

    # Volatility (ATR-based or typical range / close):
    volatility = qnta.tr(high, low, close) / close
    volatility = qnta.lwma(volatility, 14)

    # Stochastic oscillator:
    k, d = qnta.stochastic(high, low, close, 14)

    # RSI
    rsi = qnta.rsi(close)

    # Log of the closing price:
    price_log = np.log(close.ffill("time").bfill("time").fillna(0))

    # --- New Features ---
    # Bollinger Bands
    boll_up, boll_mid, boll_low = qnta.bollinger_bands(close)

    # On Balance Volume (OBV)
    obv = qnta.obv(close, vol)

    # ADX
    plus_di, minus_di, adx_val = qnta.adx(high, low, close)

    # Combine features:
    result = xr.concat(
        [
            trend,
            macd_signal,
            volatility,
            d,       # from Stochastic
            rsi,
            price_log,
            boll_up,
            boll_mid,
            boll_low,
            obv,
            plus_di,
            minus_di,
            adx_val
        ],
        pd.Index(
            [
                "trend",
                "macd_signal",
                "volatility",
                "stochastic_d",
                "rsi",
                "price_log",
                "boll_up",
                "boll_mid",
                "boll_low",
                "obv",
                "plus_di",
                "minus_di",
                "adx_val"
            ],
            name="field"
        )
    )

    return result.transpose("time", "field", "asset")

##############################################################################
#                       3) BUILD TARGET CLASSES                              #
##############################################################################

def get_target_classes(data):
    """
    Target classes for predicting if price goes up or down on the next day.
    """
    price_current = data.sel(field="close")
    price_future  = qnta.shift(price_current, -1)

    class_positive = 1  # price goes up
    class_negative = 0  # price goes down

    target_price_up = xr.where(price_future > price_current, class_positive, class_negative)
    return target_price_up

##############################################################################
#               4) MODEL: RANDOM FOREST (CHANGED FROM BAYESIAN RIDGE)        #
##############################################################################

def get_model():
    """
    Constructor for the ML model:
    Changed from BayesianRidge to RandomForestClassifier to capture
    non-linear relationships and interactions among features.
    """
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=5,
        random_state=42
    )
    return model

##############################################################################
#                    TRAIN THE MODEL (GLOBAL EXAMPLE)                        #
#        (This part is purely illustrative to show how training might go)    #
##############################################################################

my_features = get_features(stock_data)
my_targetclass = get_target_classes(stock_data)

asset_name_all = stock_data.coords["asset"].values
models = dict()

for asset_name in asset_name_all:
    features_cur = my_features.sel(asset=asset_name).dropna("time", "any")
    target_cur   = my_targetclass.sel(asset=asset_name).dropna("time", "any")

    # Align features and target by intersection of time
    target_for_learn_df, feature_for_learn_df = xr.align(target_cur, features_cur, join="inner")

    if len(features_cur.time) < 10:
        # not enough data points
        continue

    model = get_model()
    try:
        model.fit(feature_for_learn_df.values, target_for_learn_df.values)
        models[asset_name] = model
    except:
        logging.exception(f"Model training failed for {asset_name}")

##############################################################################
#                     USE THE MODEL FOR PREDICTIONS                          #
##############################################################################

weights = xr.zeros_like(stock_data.sel(field="close"))

for asset_name in asset_name_all:
    if asset_name in models:
        model = models[asset_name]
        features_cur = my_features.sel(asset=asset_name).dropna("time","any")
        if len(features_cur.time) < 1:
            continue
        try:
            # Random Forest outputs class probabilities or classes; 
            # we can use 'predict_proba' or 'predict'.
            # For a simple approach, let's take predicted class * 2 - 1 (or just the class) 
            # to get [-1, +1], but let's keep it simple as 0 or 1 for demonstration:
            pred = model.predict(features_cur.values)
            
            # convert [0,1] signals to [-1,1] weighting if desired:
            # weights.loc[dict(asset=asset_name, time=features_cur.time.values)] = 2*pred - 1 
            # or just keep them as 0 to 1:
            weights.loc[dict(asset=asset_name, time=features_cur.time.values)] = pred

        except:
            logging.exception(f"Model prediction failed for {asset_name}")

##############################################################################
#                   COMPUTE SHARPE (FOR ILLUSTRATION)                        #
##############################################################################

def get_sharpe(stock_data, weights):
    rr = qnstats.calc_relative_return(stock_data, weights)
    sharpe = qnstats.calc_sharpe_ratio_annualized(rr).values[-1]
    return sharpe

sharpe_ratio = get_sharpe(stock_data, weights)
print("Sharpe Ratio (not on rolling basis; may be forward looking):", sharpe_ratio)

##############################################################################
#            5) BACKTEST (NO FORWARD LOOKING) with QUANTIACS BACKTESTER      #
##############################################################################

def train_model(data):
    """
    This function will be called by the backtester on a rolling basis
    using only the 'past' data (no forward-looking).
    """
    asset_list = data.coords["asset"].values
    feat = get_features(data)
    tgt  = get_target_classes(data)

    local_models = {}
    for asset in asset_list:
        x = feat.sel(asset=asset).dropna("time", "any")
        y = tgt.sel(asset=asset).dropna("time", "any")

        x_aligned, y_aligned = xr.align(x, y, join="inner")

        if len(x_aligned.time) < 10:
            continue

        model = get_model()
        try:
            model.fit(x_aligned.values, y_aligned.values)
            local_models[asset] = model
        except:
            logging.exception(f"Train failed on {asset}")

    return local_models

def predict_weights(models, data):
    """
    The function to predict weights using the trained models.
    """
    asset_list = data.coords["asset"].values
    w = xr.zeros_like(data.sel(field="close"))

    feat = get_features(data)
    for asset in asset_list:
        if asset in models:
            model = models[asset]
            x = feat.sel(asset=asset).dropna("time","any")
            if len(x.time) == 0:
                continue
            try:
                pred = model.predict(x.values)
                # We can leave predictions as 0/1, or transform into [-1,1].
                w.loc[dict(asset=asset, time=x.time.values)] = pred 
            except:
                logging.exception(f"Predict failed on {asset}")

    return w

# Rolling backtest (this is the official approach to avoid forward-looking bias):
weights_rolling = qnbt.backtest_ml(
    train=train_model,
    predict=predict_weights,
    train_period=2 * 365,       # 2 years training
    retrain_interval=10 * 365,  # retrain model every 10 years for demonstration
    retrain_interval_after_submit=1,
    predict_each_day=False,
    competition_type="stocks_sp500",  # CHANGED from "stocks_nasdaq100"
    lookback_period=365,
    start_date="2005-01-01",
    analyze=True,
    build_plots=True
)

# If satisfied with the results, you can finalize submission by:
# import qnt.output as qnout
# qnout.check(weights_rolling, stock_data, "stocks_sp500")
# qnout.write(weights_rolling)
