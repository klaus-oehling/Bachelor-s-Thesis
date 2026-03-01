#%%################# LIBRARIES ####################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white, het_goldfeldquandt
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from scipy.stats import f
import figure_scale as fs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import norm, gaussian_kde
import figure_scale as fs

#%%################# INPUTS ####################
path_to_save = r'C:\Users\oehli\OneDrive - Insper - Instituto de Ensino e Pesquisa\Outros\Predocs\aaa'
path_to_data = r'C:\Users\oehli\OneDrive - Insper - Instituto de Ensino e Pesquisa\TCC\Dados/FMB/Fama-MacBeth.xlsx'
sheet_name_data = 'AÇÕES'
#sheet_name_data = 'AÇÕES MES'

#%%################# FUNCTIONS ####################

# Academic formatting
def standard_figure_fromatting(figure_width = 160):
    plt.rcParams.update({
        'figure.figsize' : fs.FigureScale(units='mm', width=figure_width, aspect=7.0/(figure_width/10)),
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 0.9,
        'axes.linewidth': line_width,
    })

    mpl.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'''
    \usepackage{amsfonts}
    \usepackage{amssymb}
    '''

# Figure saving function
def savefig(name, path):
    plt.tight_layout()
    plt.savefig(rf"{path}\{name}.svg", format='svg', bbox_inches='tight')
    plt.close()


#%%################# DATA ####################
returns = pd.read_excel(path_to_data, sheet_name=sheet_name_data)

# General data cleaning
names = list(returns.columns)
names[0] = 'Date'
returns.columns = names
returns.set_index('Date', inplace=True)
returns = returns.replace(0, np.nan)
returns = returns.rename(columns={'Beta': 'CB Stance'})

# Remove outliers
returns = returns.apply(lambda col: col.clip(lower=col.quantile(0.05), upper=col.quantile(0.95)))
returns = returns.dropna(how='all', axis=1)
all_stocks = returns.columns[0:-7]

#%%################# FAMA-MACBETH REGRESSIONS ####################

allmodels_observations_crossection = pd.DataFrame()

# Varying from 1 to 4 to run the 4 models (CAPM, FF3, FF5 and FF5+Mom)
for model in range(1,5):

    if model==1:
        colss = ['const','Standard Error', 'CB Stance', 'Market']
        returns_model = returns.drop(['Size', 'Profitability','Momentum','Value','Investment'], axis=1)
        factors = 2

    elif model==2:
        colss = ['const','Standard Error','CB Stance','Market', 'Size', 'Value']
        returns_model = returns.drop(['Profitability','Momentum','Investment'], axis=1)
        factors = 4

    elif model==3:
        colss = ['const', 'Standard Error', 'CB Stance','Market', 'Size','Value', 'Investment', 'Profitability']
        returns_model = returns.drop(['Momentum'], axis=1)
        factors = 6
        
    else:
        colss = ['const', 'Standard Error', 'CB Stance', 'Market', 'Size', 'Value','Investment', 'Profitability', 'Momentum']
        returns_model = returns
        factors = 7

    # DataFrames to store coefficients, p-values and number of observations of the cross section regressions
    crosssection_coefficients = pd.DataFrame(index=returns_model.index[(factors+1)**2:], columns=colss, dtype=float)
    crosssection_pvalues = pd.DataFrame(index=returns_model.index[(factors+1)**2:], columns=colss, dtype=float)
    observations_crosssection_reg = pd.Series(index=returns_model.index[((factors+1)**2):], dtype=float)

    # DataFrame to store number of observations of the time series regressions for each stock and date
    observations_timeseries_reg = pd.DataFrame()

    for i, d in enumerate(returns_model.index[((factors+1)**2-1):-1]):
        
        # Slice returns to prevent look-ahead bias
        ret_timeseries_reg = returns_model.loc[:d]

        # Remove all_stocks with too many NaN values
        drop_cols = (np.isnan(ret_timeseries_reg)).sum() > i
        drop_list = drop_cols[drop_cols].index.tolist()
        ret_timeseries_reg = ret_timeseries_reg.drop(columns=drop_list)

        # If it's December or the first iteration, run time series regressions for all all_stocks to get betas and standard errors
        if d.month == 12 or i==0:

            # Define regressors and regressands
            regressors_timeseries_reg = ret_timeseries_reg.iloc[:, -factors:]
            regressors_timeseries_reg = sm.add_constant(regressors_timeseries_reg)
            regressands_timeseries_reg = ret_timeseries_reg.columns[:len(ret_timeseries_reg.columns)-factors]

            # DataFrames to store coefficients, p-values, standard errors and number of observations for each stock's time series regression in the current date
            coefs_df = pd.DataFrame(index=regressands_timeseries_reg, columns=regressors_timeseries_reg.columns[1:], dtype=float)
            pvalues_df = pd.DataFrame(index=regressands_timeseries_reg, columns=regressors_timeseries_reg.columns[1:], dtype=float)
            ses_df = pd.Series(index=regressands_timeseries_reg, dtype=float)
            obs_df = pd.Series(index=regressands_timeseries_reg, dtype=float)

            # Define each stocks' sensibility to the factors
            for stock in regressands_timeseries_reg:
                
                # Define X and Y for the time series regression and remove NaN values
                Y = ret_timeseries_reg[stock][:] # Specific stock
                mask = np.isnan(Y)
                idx_to_remove = Y.index[mask]
                Y = Y.loc[~mask] # Remove NaN values
                X = regressors_timeseries_reg.drop(index=idx_to_remove) # Remove corresponding Y NaN values

                # Run OLS regression and get residuals
                results = sm.OLS(Y, X).fit()
                residuals = results.resid

                # Autocorrelation test (Ljung-Box test)
                lags = int(round(12 * (len(residuals) / 100)**0.25))
                lb_test = acorr_ljungbox(residuals, lags=[lags], return_df=True)
                lb_test_pvalue = lb_test['lb_pvalue'].values[0]

                # Heteroscedasticity test (White test)
                white_test = het_white(residuals, results.model.exog)
                white_test_pvalue = white_test[3]

                # Save results
                # Only consider stocks that pass both tests (no autocorrelation and homoscedasticity)
                if lb_test_pvalue>0.05 and white_test_pvalue>0.05:
                    coefs_df.loc[stock] = results.params[1:] # Coefficients
                    pvalues_df.loc[stock] = results.pvalues[1:] # P-values
                    ses_df.loc[stock] = results.mse_resid ** 0.5 # Standard errors
                    obs_df.loc[stock] = results.nobs # Number of observations
                    
            observations_timeseries_reg = pd.concat([observations_timeseries_reg, obs_df.rename(d)], axis=1) 

            # Treat the coefficients dataframe to be used as regressors in the cross section regression
            coefs_df = pd.concat([coefs_df, ses_df.rename('Standard Error')], axis=1) # Add standard errors to the coefficients dataframe as it will be used as a regressor
            coefs_df = coefs_df.dropna(how='any') # Drop stocks that didn't pass the tests
            regressors_crosssection_reg = sm.add_constant(coefs_df)

        # The cross section regression uses the next date to prevent look-ahead bias from the regressors estimation
        all_dates = pd.Series(returns_model.index)
        date_crosssection_reg = all_dates.loc[all_dates[all_dates == d].index[0]+1]
        
        # Define X and Y for the cross section regression
        Y = returns_model.loc[date_crosssection_reg] # Specific date
        Y = Y[regressors_crosssection_reg.index] # Only consider stocks we have regressors for
        mask = np.isnan(Y)
        idx_to_remove = Y.index[mask]
        Y = Y.loc[~mask] # Remove NaN values
        X = regressors_crosssection_reg.drop(index=idx_to_remove) # Remove corresponding Y NaN values
        
        # Run OLS regression and get residuals
        results = sm.OLS(Y, X).fit()
        
        # Save results
        # Only consider cross section regressions with enough observations (square of the number of regressors [number of factors + constant + standard error])
        if results.nobs>=(factors+2)**2:
            observations_crosssection_reg.loc[date_crosssection_reg] = results.nobs
            crosssection_coefficients.loc[date_crosssection_reg] = results.params
            crosssection_pvalues.loc[date_crosssection_reg] = results.pvalues

    # Standardize and prepare the number of observations dataframes
    observations_timeseries_reg = observations_timeseries_reg.reindex(all_stocks)
    observations_crosssection_reg = observations_crosssection_reg.dropna()
    allmodels_observations_crossection = pd.concat([allmodels_observations_crossection, observations_crosssection_reg], axis=1)

    #%%################# RESULTS ####################
    degrees_freedom = round(sum(observations_crosssection_reg)/len(observations_crosssection_reg),0)-(factors+2)

    # Calculate average coefficients, standard errors and t-stats
    average_coefficients = crosssection_coefficients.mean(axis=0, skipna=True)
    coefficients_standarderrors = crosssection_coefficients.std(ddof=1) / (np.sqrt(crosssection_coefficients.dropna().shape[0]))
    coefficients_tstats = average_coefficients/coefficients_standarderrors

    # Calculate p-values for the average coefficients using the t-distribution
    crosssection_pvalues = pd.DataFrame(2 * stats.t.sf(coefficients_tstats.abs(), df=degrees_freedom))
    crosssection_pvalues.index = coefficients_tstats.index

    # Combine results, change labels and print
    results_df = pd.concat([average_coefficients, coefficients_standarderrors, coefficients_tstats, crosssection_pvalues], axis=1)
    results_df.columns = ['Average Coefficients', 'Standard Errors', 'T-Stats', 'P-Values']
    new_index = results_df.index.to_list()
    new_index[0] = 'Constant'
    results_df.index = new_index
    print(round(results_df,5))

    #%%################# CHARTS ####################

    # Settings
    line_width = 0.7
    standard_figure_fromatting(figure_width = 80)
    lower_limit=0
    upper_limit=80
    steps=10
    #lower_limit=0
    #upper_limit=240
    #steps=30

    # Ploting
    for col in observations_timeseries_reg.columns:
        plt.scatter(observations_timeseries_reg.index, observations_timeseries_reg[col], s=3, marker='o', facecolors='#1f77b4', edgecolors='none')
    plt.axhline((factors+1)**2, color='black', linewidth=0.7, linestyle='--')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(observations_timeseries_reg.index[0], observations_timeseries_reg.index[-1])
    plt.ylim(lower_limit,upper_limit)
    plt.yticks(range(lower_limit,upper_limit+steps,steps))
    tick_positions = np.arange(0, len(observations_timeseries_reg), 30)
    plt.xticks(tick_positions, [''] * len(tick_positions))
    plt.xlabel('Stocks')
    plt.ylabel('Observations')
    savefig(name = f'Time Series Observations model {model}', path = path_to_save)

# Settings
line_width = 0.7
standard_figure_fromatting(figure_width = 160)
lower_limit=0
upper_limit=350
steps=50
legend_location = 'lower left'
#legend_location = 'lower right'

# Ploting
plt.figure()
plt.plot(allmodels_observations_crossection.iloc[:,0], color='#1f77b4')
plt.plot(allmodels_observations_crossection.iloc[:,1], color="black")
plt.plot(allmodels_observations_crossection.iloc[:,2], color="green")
plt.plot(allmodels_observations_crossection.iloc[:,3], color="grey")
plt.grid(True, linestyle='--', alpha=0.6)

plt.axhline((4)**2, color='#1f77b4', linewidth=line_width, linestyle='--')
plt.axhline((6)**2, color='black', linewidth=line_width, linestyle='--')
plt.axhline((8)**2, color='green', linewidth=line_width, linestyle='--')
plt.axhline((9)**2, color='grey', linewidth=line_width, linestyle='--')

plt.xlim(allmodels_observations_crossection.index.min(), allmodels_observations_crossection.index.max())
plt.ylim(lower_limit,upper_limit)
plt.yticks(range(lower_limit,upper_limit+steps,steps))
plt.legend(['CAPM','FF3','FF5','FF5+Mom'], loc=legend_location)
plt.ylabel('Observations')
savefig(name = 'Cross Section Observations all models', path = path_to_save)
