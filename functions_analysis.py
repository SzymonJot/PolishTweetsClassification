import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

def fit_regressions(df, stock_columns, tweet_columns):
    results = []
    for stock_var in stock_columns:
        for tweet_var in tweet_columns:
            print(tweet_var)
            X = df[[tweet_var]]
            y = df[stock_var]
            
            valid_data = pd.concat([X, y], axis=1).dropna()
            X = valid_data[[tweet_var]]
            y = valid_data[stock_var]
            
            # Remove outliers using z-score method
            z_scores = np.abs(stats.zscore(X))
            outlier_mask = (z_scores < 3).all(axis=1)
            X = X[outlier_mask]
            y = y[outlier_mask]
            
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            
            # Diagnostic tests
            _, normality_pvalue = stats.normaltest(model.resid)
            _, homoscedasticity_pvalue = stats.levene(X[tweet_var], model.resid)
            durbin_wat = durbin_watson(model.resid)
            vif = variance_inflation_factor(X.values, 1)
            
            results.append({
                'Stock Variable': stock_var,
                'Tweet Variable': tweet_var,
                'R-squared': model.rsquared.round(3),
                'Adj R-squared': model.rsquared_adj.round(3),
                'Coefficient': model.params[tweet_var].round(3),
                'P-value': model.pvalues[tweet_var].round(5),
                'Residuals Normality P-value': normality_pvalue.round(5),
                'Homoscedasticity P-value': homoscedasticity_pvalue.round(5),
                'Durbin-Watson': durbin_wat.round(3),
                'VIF': vif.round(3),
                'Sample Size': len(X),
                'Outliers Removed': len(valid_data) - len(X)
            })
    
    return pd.DataFrame(results)