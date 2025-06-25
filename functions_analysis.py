import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from itertools import combinations
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


def find_best_combination(df, stock_column, tweet_columns, max_features=3):
    best_result = {
        'R-squared': 0,
        'Combination': None,
        'Model': None
    }

    for n in range(1, min(max_features, len(tweet_columns)) + 1):
        for combo in combinations(tweet_columns, n):
            X = df[list(combo)]
            y = df[stock_column]
            
            valid_data = pd.concat([X, y], axis=1).dropna()
            X = valid_data[list(combo)]
            y = valid_data[stock_column]

            z_scores = np.abs(stats.zscore(X))
            outlier_mask = (z_scores < 3).all(axis=1)
            X = X[outlier_mask]
            y = y[outlier_mask]
            
            
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            _, normality_pvalue = stats.normaltest(model.resid)
            
            if model.rsquared > best_result['R-squared']:
                best_result = {
                    'R-squared': model.rsquared,
                    'Combination': combo,
                    'Model': model,
                    'Residuals Normality P-value': normality_pvalue
                }
    
    return best_result

def analyze_stocks(df, stock_columns, tweet_columns, max_features=3):
    results = []
    
    for stock_var in stock_columns:
        print(f"Analyzing {stock_var}...")
        best = find_best_combination(df, stock_var, tweet_columns, max_features)
        
        result = {
            'Stock Variable': stock_var,
            'Best Tweet Varaibles Combination': best['Combination'],
            'R-squared': best['R-squared'].round(3),
            'Residuals Normality P-value': best['Residuals Normality P-value']
        }
        
        for feature in best['Combination']:
            result[f'Coefficient_{feature}'] = best['Model'].params[feature].round(10)
            result[f'P-value_{feature}'] = best['Model'].pvalues[feature].round(5)
        
        results.append(result)
    
    return results

all_results = {}


def create_prediction_lags(df, tweet_col=[], max_lags=5):
    trading_days = df[df['is_trading_day']].copy().reset_index(drop=True)
    
    for col in tweet_col:
        for lag in range(1, max_lags + 1):
            lagged_values = []
            
            for idx, row in trading_days.iterrows():
                current_date = row['Date']
                target_date = current_date - pd.Timedelta(days=lag)
                
                # Get tweet for exact target date
                tweet_row = df[df['Date'] == target_date]
                
                if len(tweet_row) > 0 and not pd.isna(tweet_row[col].iloc[0]):
                    lagged_values.append(tweet_row[col].iloc[0])
                else:
                    lagged_values.append(np.nan)
            
            trading_days[f'{col}_lag_{lag}'] = lagged_values
    
    return trading_days

def flatten_granger_results(granger_dict):
    rows = []
    for company, company_dict in granger_dict.items():
        for target, target_dict in company_dict.items():
            for source, source_dict in target_dict.items():
                for lag_feature, feature_p in source_dict.items():
                    rows.append({
                            "Company": company,
                            "Target Variable": target,
                            "Source Feature": source,
                            "Lag": lag_feature,
                            "P-Value": feature_p
                        })
    return pd.DataFrame(rows)