import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler

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


def block_shuffle(series, block_size):
    """Shuffle blocks of a time series while preserving internal structure."""
    n = len(series)
    n_blocks = n // block_size
    blocks = [series[i*block_size:(i+1)*block_size].values for i in range(n_blocks)]
    
    # Handle remainder
    remainder = n % block_size
    if remainder > 0:
        blocks.append(series[n_blocks*block_size:].values)
    
    np.random.shuffle(blocks)
    return pd.Series(np.concatenate(blocks), index=series.index)


def select_lags(stock_var, tweet_vars, max_lags=5):
    """Choose optimal lags via mutual information with stock variable"""
    mi_scores = {}
    for lag in range(1, max_lags+1):
        lagged_vars = [f"{var}_lag_{lag}" for var in tweet_vars]
        X = data[lagged_vars]
        mi = mutual_info_regression(X, data[stock_var])
        mi_scores[lag] = np.mean(mi)
        
    # Keep top 2 lags with highest MI
    best_lags = sorted(mi_scores, key=mi_scores.get, reverse=True)[:2]
    return [f"{var}_lag_{l}" for l in best_lags for var in tweet_vars]


def test_granger_bootstrap(df, stock_var, twitter_var, maxlag, B, bootstrap_block_size):
    # Reset index to avoid label mismatch issues
    df = df.reset_index(drop=True).copy()
    dat = []
    significant_lags = {}
    base_cols = [stock_var] + [twitter_var]

    data = df[base_cols].copy()
    boot_res = []
    
    # Generate stock lags
    for lag in range(1, maxlag+1):
        data[f"{stock_var}_lag_{lag}"] = data[stock_var].shift(lag)
        data[f"{twitter_var}_lag_{lag}"] = data[twitter_var].shift(lag)
    
    # Test each lag order
    for p in range(1, maxlag+1):
        # Prepare data for current lag p
        required_cols = [stock_var]
        stock_lags = [f"{stock_var}_lag_{i}" for i in range(1, p+1)]
        twitter_lags = [f"{twitter_var}_lag_{i}" for i in range(1, p+1)]
        
        test_data = data[required_cols + stock_lags + twitter_lags].dropna()
        n_obs = len(test_data)
        if n_obs < 2*p + 5:
            continue
        
        # Split data
        y = test_data[stock_var]
        X_restricted = test_data[stock_lags]
        X_unrestricted = test_data[stock_lags + twitter_lags]
        
        # Add constants
        X_restricted = sm.add_constant(X_restricted)
        X_unrestricted = sm.add_constant(X_unrestricted)
        
        try:
            # Fit original models
            model_restricted = sm.OLS(y, X_restricted).fit()
            model_unrestricted = sm.OLS(y, X_unrestricted).fit()
            
            # Calculate actual F-statistic
            rss_r = model_restricted.ssr
            rss_ur = model_unrestricted.ssr
            k = len(twitter_lags)  # Extra parameters
            dfd = n_obs - X_unrestricted.shape[1]  # Denominator DF
            
            if dfd <= 0:
                continue
            
            F_actual = ((rss_r - rss_ur) / k) / (rss_ur / dfd)
            
            # Block bootstrap procedure
            count_exceeds = 0
            n_valid = 0
            idx = test_data.index
            
            # Initialize bootstrap results for this lag order
            boot_res_p = []
            
            for i in range(B):
                # Create block-shuffled Twitter series
                boot_twitter = block_shuffle(df[twitter_var], block_size=bootstrap_block_size)
                
                # Rebuild Twitter lags from shuffled series
                boot_twitter_lags = pd.DataFrame(index=df.index)
                for lag in range(1, p+1):
                    boot_twitter_lags[f"{twitter_var}_lag_{lag}"] = boot_twitter.shift(lag)
                
                # Align bootstrap data with test_data index
                boot_selected = boot_twitter_lags.loc[idx, twitter_lags].reset_index(drop=True)
                stock_selected = test_data[stock_lags].reset_index(drop=True)
                
                # Combine with original stock data
                X_unrestricted_boot = pd.concat([stock_selected, boot_selected], axis=1)
                
                X_unrestricted_boot = sm.add_constant(X_unrestricted_boot)
                
                # Skip iterations with insufficient data
                if X_unrestricted_boot.isnull().any().any():
                    continue
                
                try:
                    # Refit model with shuffled Twitter data
                    model_boot = sm.OLS(y.reset_index(drop=True), X_unrestricted_boot).fit()
                    rss_ur_boot = model_boot.ssr
                    
                    # Calculate bootstrap F-statistic
                    # FIXED: Use correct RSS values for F-statistic
                    F_boot = ((rss_r - rss_ur_boot) / k) / (rss_ur_boot / dfd)
                    boot_res_p.append(F_boot)
                    
                    if F_boot >= F_actual:
                        count_exceeds += 1
                    n_valid += 1
                  
                except Exception as e:
                    continue
            
            # Calculate bootstrap p-value
            p_value = count_exceeds / n_valid if n_valid > 0 else np.nan
            significant_lags[p] = p_value
            
            # Store bootstrap results for this specific lag order
            boot_res.append({
                'lag_order': p,
                'F_actual': F_actual,
                'boot_F_stats': boot_res_p,
                'p_value': p_value,
                'n_valid': n_valid,
                'count_exceeds': count_exceeds
            })
            
        except Exception as e:
            print(f"Error processing lag {p}: {str(e)}")
            continue
    
    return {
        'significant_lags': significant_lags,
        'boot_res': boot_res
    }



def winsorize_series(series, lower_pct, upper_pct):
    """Winsorize a pandas Series or numpy array"""
    # Convert to numpy array if needed
    if isinstance(series, pd.Series):
        values = series.values
    else:
        values = series
        
    lower_bound = np.percentile(values, lower_pct)
    upper_bound = np.percentile(values, upper_pct)
    
    # Clip the values
    winsorized = np.clip(values, lower_bound, upper_bound)
    
    return winsorized, lower_bound, upper_bound


def winsorize_dataframes(train_df, val_df, test_df, lower_pct=5, upper_pct=95):
    train_win = train_df.copy()
    val_win = val_df.copy()
    test_win = test_df.copy()
    bounds = {}
    
    for col in train_df.columns:
        # Winsorize training data
        win_vals, low, high = winsorize_series(
            train_df[col], lower_pct, upper_pct
        )
        train_win[col] = win_vals
        bounds[col] = (low, high)
        
        # Apply to validation/test
        val_win[col] = np.clip(val_df[col], low, high)
        test_win[col] = np.clip(test_df[col], low, high)
        
    return train_win, val_win, test_win, bounds

def simple_train_valid_test_split_standardized(data, train_ratio=0.7, valid_ratio=0.1):
    """
    Split data chronologically into train, validation, and test sets,
    and standardize features based on the training set only.
    Handles only numeric columns for scaling.
    """
    n = len(data)
    train_end = int(n * train_ratio)
    valid_end = train_end + int(n * valid_ratio)
    
    if train_end == 0 or valid_end <= train_end or valid_end >= n:
        raise ValueError(
            f"Invalid split ratios with data of length {n}: "
            f"train_ratio={train_ratio}, valid_ratio={valid_ratio}."
        )
    
    train = data.iloc[:train_end]
    valid = data.iloc[train_end:valid_end]
    test = data.iloc[valid_end:]
    
    if len(test) == 0:
        raise ValueError("Test set is empty. Adjust your split ratios.")
    
    numeric_cols = data.select_dtypes(include='number').columns

    scaler = StandardScaler()
    scaler.fit(train[numeric_cols])

    train_scaled = pd.DataFrame(scaler.transform(train[numeric_cols]), index=train.index, columns=numeric_cols)
    valid_scaled = pd.DataFrame(scaler.transform(valid[numeric_cols]), index=valid.index, columns=numeric_cols)
    test_scaled = pd.DataFrame(scaler.transform(test[numeric_cols]), index=test.index, columns=numeric_cols)
    
    return train_scaled, valid_scaled, test_scaled, scaler
