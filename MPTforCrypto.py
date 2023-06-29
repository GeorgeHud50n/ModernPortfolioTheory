
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# Define the list of cryptocurrencies to include in the portfolio
crypto_list = ['BTC-GBP', 'ETH-GBP', 'SOL-GBP','DOT-GBP','LTC-GBP','ATOM-GBP','ADA-GBP', 'XRP-GBP', 'MATIC-GBP', 'HBAR-GBP', 'XLM-GBP', 'ICP-GBP']

# Fetch historical price data
start_date = '2020-01-01'
end_date = '2023-04-25'

prices = yf.download(crypto_list, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
daily_returns = prices.pct_change().dropna()

# Calculate mean returns and covariance matrix
mean_daily_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()

# Calculate the negative Sharpe Ratio
def sharpe_ratio_neg(weights, mean_daily_returns, cov_matrix, risk_free_rate):
    portfolio_return = np.dot(weights, mean_daily_returns) * 365
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(365)
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    return -sharpe_ratio

# Find the optimal portfolio for each crypto
optimal_weights_list = []
sharpe_ratio_list = []

for crypto in crypto_list:
    single_crypto_list = [crypto]
    single_crypto_mean_daily_returns = mean_daily_returns[single_crypto_list]
    single_crypto_cov_matrix = cov_matrix.loc[single_crypto_list, single_crypto_list]
    
    result = minimize(sharpe_ratio_neg, [1], args=(single_crypto_mean_daily_returns, single_crypto_cov_matrix, 0.02), bounds=[(0, 1)])
    optimal_weight = result.x
    sharpe_ratio_value = -sharpe_ratio_neg(optimal_weight, single_crypto_mean_daily_returns, single_crypto_cov_matrix, 0.02)
    
    optimal_weights_list.append(optimal_weight)
    sharpe_ratio_list.append(sharpe_ratio_value)

# Sort the cryptos by their Sharpe ratio and select the top 5
sorted_cryptos = sorted(zip(crypto_list, sharpe_ratio_list), key=lambda x: x[1], reverse=True)
best_5_cryptos = [crypto for crypto, sharpe in sorted_cryptos[:5]]

# Perform the Efficient Frontier analysis for the selected 5 cryptos
best_5_prices = prices[best_5_cryptos]
best_5_daily_returns = best_5_prices.pct_change().dropna()
best_5_mean_daily_returns = best_5_daily_returns.mean()
best_5_cov_matrix = best_5_daily_returns.cov()

# Set initial investment
initial_investment = 5000

# Generate random portfolio weights
num_portfolios = 10000
weights_matrix = np.random.random((num_portfolios, len(best_5_cryptos)))
weights_matrix = weights_matrix / np.sum(weights_matrix, axis=1)[:, np.newaxis]

# Calculate portfolio returns and risks
portfolio_returns = np.dot(weights_matrix, best_5_mean_daily_returns) * 365
portfolio_risks = np.sqrt(np.diag(np.dot(np.dot(weights_matrix, best_5_cov_matrix), weights_matrix.T))) * np.sqrt(365)

# Calculate the Sharpe Ratio
risk_free_rate = 0.02
sharpe_ratio = (portfolio_returns - risk_free_rate) / portfolio_risks

# Find the optimal portfolio
optimal_portfolio_index = np.argmax(sharpe_ratio)
optimal_weights = weights_matrix[optimal_portfolio_index]

# Define a function to find the minimum volatility for a given target return
def find_min_volatility(target_return, mean_daily_returns, cov_matrix):
    def objective_function(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(365)

    def target_return_constraint(weights, mean_daily_returns, target_return):
        return np.dot(weights, mean_daily_returns) * 365 - target_return

    constraints = [{'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                   {'type': 'eq', 'fun': lambda weights: target_return_constraint(weights, mean_daily_returns, target_return)}]
    bounds = [(0, 1) for _ in range(len(best_5_cryptos))]
    initial_guess = [1 / len(best_5_cryptos)] * len(best_5_cryptos)
    result = minimize(objective_function, initial_guess, args=(cov_matrix), bounds=bounds, constraints=constraints)

    min_volatility = objective_function(result.x, cov_matrix)
    return min_volatility

# Generate a range of target returns and find the minimum volatility for each
target_returns = np.linspace(portfolio_returns.min(), portfolio_returns.max(), 100)
min_volatilities = [find_min_volatility(tr, best_5_mean_daily_returns, best_5_cov_matrix) for tr in target_returns]

# Plot Efficient Frontier
plt.figure(figsize=(12, 6))
plt.scatter(portfolio_risks, portfolio_returns, c=sharpe_ratio, cmap='viridis', marker='o', s=10, alpha=0.3)
plt.plot(min_volatilities, target_returns, 'r-', linewidth=2, label='Efficient Frontier')
plt.colorbar(label='Sharpe Ratio')
plt.scatter(portfolio_risks[optimal_portfolio_index], portfolio_returns[optimal_portfolio_index], c='red', marker='*', s=300)
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.legend()
plt.show()

# Print optimal portfolio weights
optimal_portfolio = pd.DataFrame(data=optimal_weights, index=best_5_cryptos, columns=['Weight'])
optimal_portfolio['Investment'] = optimal_portfolio['Weight'] * initial_investment
print("Optimal Portfolio Weights:\n", optimal_portfolio)

################## RISK-MANAGEMENT ###############################################################
# Calculate portfolio daily returns using the optimal weights
optimal_portfolio_returns = best_5_daily_returns.dot(optimal_weights)

# Compute the historical 1-day VaR at a 95% confidence level
confidence_level = 0.95
VaR = -np.percentile(optimal_portfolio_returns, 100 * (1 - confidence_level))
VaR_investment = initial_investment * VaR

# Calculate the Expected Shortfall (ES) at a 95% confidence level
ES = -optimal_portfolio_returns[optimal_portfolio_returns < -VaR].mean()
ES_investment = initial_investment * ES

print(f"1-Day VaR (95% confidence level): {VaR:.4f} or £{VaR_investment:.2f}")
print(f"1-Day Expected Shortfall (95% confidence level): {ES:.4f} or £{ES_investment:.2f}")
