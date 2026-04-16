import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.optimize import minimize

RISK_FREE_RATE: float = 0.05

# Filter thresholds for option data cleaning
MIN_VOLUME: int = 1
MAX_REL_SPREAD: float = 0.50 # Maximum relative spread (Spread / MidPrice)
MIN_DAYS_TO_EXPIRY: int = 1 # Minimum days to expiry to include
MIN_IV: float = 0.01 # Minimum plausible IV
MAX_IV: float = 100.00 # Maximum plausible IV
# Interpolation Grid Resolution
N_STRIKES_GRID: int = 50
N_EXPIRIES_GRID: int = 50

def black_scholes_price(
    S: float, K: float, T: float, r: float, sigma: float
) -> float:
    """
    Calculates the Black-Scholes price for a European option.

    Args:
        S: Current stock price.
        K: Option strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        sigma: Volatility of the underlying stock (annualized).

    Returns:
        The Black-Scholes price of the option. Returns NaN if inputs are invalid.
    """
    if sigma <= 0 or T <= 0:
        return max(0.0, S - K * np.exp(-r * T))

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    except OverflowError:
         return np.nan

    return max(0.0, price)

def implied_volatility(
    S: float, K: float, T: float, r: float, market_price: float,
    low_vol: float = 1e-4, high_vol: float = 4.0, tol: float = 1e-6
) -> float:
    """
    Calculates the implied volatility using Brent's root-finding method.

    Args:
        S: Current stock price.
        K: Option strike price.
        T: Time to expiration in years.
        r: Risk-free interest rate (annualized).
        market_price: The observed market price of the option.
        option_type: 'call' or 'put'.
        low_vol: Lower bound for volatility search.
        high_vol: Upper bound for volatility search.
        tol: Tolerance for the root-finding algorithm.

    Returns:
        The implied volatility (annualized), or np.nan if calculation fails or
        market price violates arbitrage bounds.
    """
    if T <= 0 or market_price <= 0:
        return np.nan


    # Check arbitrage violations
    if market_price < max(0.0, S - K * np.exp(-r * T) - tol):
         print(f"Price violation: Call price {market_price:.4f} < Intrinsic {max(0.0, S - K * np.exp(-r * T)):.4f}")
         return np.nan

     # Check maximum price bounds
    if market_price > S: # Call price cannot exceed stock price
        print(f"Price violation: Call price {market_price:.4f} > Stock Price {S:.4f}")
        return np.nan


    # objective function for root finder
    def objective_func(sigma: float) -> float:
        # Return large value for invalid sigma to guide solver
        if sigma <= 0:
            return 1e10
        try:
            model_price = black_scholes_price(S, K, T, r, sigma)
            # Check if model price is NaN (can happen from black_scholes_price)
            if np.isnan(model_price):
                 return 1e11 # Indicate error
            return model_price - market_price
        except (ValueError, OverflowError):
            return 1e12 # Indicate numerical error

    # Attempt to bracket the root
    try:
        f_low = objective_func(low_vol)
        f_high = objective_func(high_vol)

        if f_low > 1e9 or f_high > 1e9:
            print(f"Objective function error at bounds for K={K}, T={T:.4f}")
            return np.nan

        price_at_low_vol = black_scholes_price(S, K, T, r, low_vol)
        price_at_high_vol = black_scholes_price(S, K, T, r, high_vol)

        if np.isnan(price_at_low_vol) or np.isnan(price_at_high_vol):
            print(f"BS price calculation failed at bounds for K={K}, T={T:.4f}")
            return np.nan

        # If market price is below the price at min vol (and above intrinsic), IV might be < low_vol
        if market_price < price_at_low_vol - tol:
            print(f"Market price {market_price:.4f} below price at low vol bound {price_at_low_vol:.4f} for K={K}, T={T:.4f}")
            return np.nan
        # If market price is above the price at max vol
        if market_price > price_at_high_vol + tol:
             print(f"Market price {market_price:.4f} above price at high vol bound {price_at_high_vol:.4f} for K={K}, T={T:.4f}")
             return np.nan


        # Check if signs are different (required for brentq)
        if np.sign(f_low) == np.sign(f_high):
            # Try adjusting bounds slightly if signs are same and price is within range
             if abs(f_low) < abs(f_high):
                  high_vol_adj = high_vol * 1.5
                  f_high_adj = objective_func(high_vol_adj)
                  if np.sign(f_low) != np.sign(f_high_adj):
                      high_vol = high_vol_adj
                  else:
                       print(f"Cannot bracket root (sign issue) K={K}, T={T:.4f}. f({low_vol:.4f})={f_low:.4e}, f({high_vol:.4f})={f_high:.4e}")
                       return np.nan
             else:
                  low_vol_adj = low_vol * 0.5
                  f_low_adj = objective_func(low_vol_adj)
                  if np.sign(f_low_adj) != np.sign(f_high):
                      low_vol = low_vol_adj
                  else:
                      print(f"Cannot bracket root (sign issue) K={K}, T={T:.4f}. f({low_vol:.4f})={f_low:.4e}, f({high_vol:.4f})={f_high:.4e}")
                      return np.nan


    except (ValueError, OverflowError):
         print(f"Numerical error during bound check K={K}, T={T:.4f}")
         return np.nan


    # Use Brent's method for root finding
    try:
        iv = brentq(objective_func, low_vol, high_vol, xtol=tol, rtol=tol)
    except ValueError:
        print(f"Brentq ValueError K={K}, T={T:.4f}. Check bounds and objective function.")
        return np.nan
    except Exception as e:
        print(f"Unexpected error in brentq K={K}, T={T:.4f}: {e}")
        return np.nan

    if not (low_vol <= iv <= high_vol):
        
        print(f"Warning: Calculated IV {iv:.4f} outside search range [{low_vol:.4f}, {high_vol:.4f}] for K={K}, T={T:.4f}")
        return np.nan

    return iv

def calculate_iv_for_chain(
    option_chain_df: pd.DataFrame,
    r: float,
    min_volume: int = MIN_VOLUME,
    max_rel_spread: float = MAX_REL_SPREAD,
    min_days_expiry: int = MIN_DAYS_TO_EXPIRY
) -> pd.DataFrame:
    """
    Calculates Implied Volatility for a filtered option chain DataFrame.

    Args:
        option_chain_df: DataFrame from fetch_option_chain.
        r: Risk-free interest rate.
        min_volume: Minimum trading volume to include the option.
        max_rel_spread: Maximum relative bid-ask spread allowed.
        min_days_expiry: Minimum number of days to expiry.

    Returns:
        DataFrame with 'TimeToExpiry', 'Strike', 'ImpliedVolatility', 'Type'.
        Filters out options failing criteria or IV calculation.
    """
    df = option_chain_df.copy()

    # Ensure required columns exist
    required_cols = ['best_bid', 'best_ask', 'strike', 'time_to_maturity', 'fetch_date', 'spot_price']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error: Missing required columns in input DataFrame: {missing}")
        # Return structure consistent with success, but empty
        return pd.DataFrame(columns=['time_to_maturity', 'strike'])


    # Calculate Mid Price and Time to Expiry
    df['MidPrice'] = (df['best_bid'] + df['best_ask']) / 2.0

    # --- Filtering ---
    # 1. Price and Spread validity
    df = df[df['best_bid'] > 0]
    df = df[df['best_ask'] > 0]
    df = df[df['MidPrice'] > 0]
    df['Spread'] = df['best_ask'] - df['best_bid']
    df['RelativeSpread'] = df['Spread'] / df['MidPrice']
    df = df[df['RelativeSpread'] <= max_rel_spread]
    df = df[df['RelativeSpread'] >= 0] # Ensure spread is not negative


    # 3. Time to Expiry
    df = df[df['time_to_maturity'] * 365.25 >= min_days_expiry]
    df = df[df['time_to_maturity'] > 1e-6] # Avoid zero or negative time

    # 4. Drop rows with NaN in critical columns before IV calculation
    critical_cols_iv = ['spot_price', 'strike', 'time_to_maturity', 'MidPrice']
    df.dropna(subset=critical_cols_iv, inplace=True)

    # Filter out options where mid-price is clearly below intrinsic value (arbitrage)
    df['IntrinsicValue'] = np.maximum(0.0, df['spot_price'] - df['strike'] * np.exp(-r * df['time_to_maturity']))
    # Allow a small tolerance for market friction / minor price discrepancies
    df = df[df['MidPrice'] >= df['IntrinsicValue'] - 1e-6] # Price should not be significantly below intrinsic


    if df.empty:
         print("Warning: No valid options remaining after initial filtering.")
         return pd.DataFrame(columns=['time_to_maturity', 'strike'])

    # Calculate IV row-wise (Vectorization is complex due to root finding)
    # Consider using joblib or multiprocessing for large chains if performance is critical
    iv_values = []
    for _, row in df.iterrows():
        iv = implied_volatility(
            S=row['spot_price'],
            K=row['strike'],
            T=row['time_to_maturity'],
            r=r,
            market_price=row['MidPrice'],
        )
        iv_values.append(iv)

    df['ImpliedVolatility'] = iv_values

    # --- Post-IV Filtering ---
    # 1. Remove rows where IV calculation failed (returned NaN)
    df.dropna(subset=['ImpliedVolatility'], inplace=True)

    # 2. Filter based on plausible IV range
    df = df[(df['ImpliedVolatility'] >= MIN_IV) & (df['ImpliedVolatility'] <= MAX_IV)]

    # Select and rename final columns
    result_df = df[['time_to_maturity', 'strike', 'ImpliedVolatility']].copy()

    return result_df

def svi_raw_total_variance(k, a, b, rho, m, sigma):
    y = k - m
    return a + b * (rho * y + np.sqrt((k - m)**2 + sigma**2))

def svi_raw_first_derivative(k, a, b, rho, m, sig):
    y = k - m
    return b * (rho + y / np.sqrt(y**2 + sig**2))

def svi_raw_second_derivative(k, a, b, rho, m, sig):
    y = k - m
    return (b * sig**2) / (y**2 + sig**2)**(1.5)

def svi_iv_from_params(K, params):
    """Convert raw-SVI total variance back to Black-Scholes IV at strikes K."""
    K = np.asarray(K, dtype=float)
    F, T = params["F"], params["T"]
    k = np.log(K / F)
    w = svi_raw_total_variance(k, params["a"], params["b"], params["rho"], params["m"], params["sigma"])
    w = np.maximum(w, 0.0)
    return np.sqrt(w / T)

def fit_svi_raw_one_slice(df_slice, F, prev_params=None, F_prev=None, K_grid_cons=None, 
                          T_col="time_to_maturity", K_col="strike", iv_col="ImpliedVolatility"):
    
    T = float(df_slice[T_col].iloc[0])
    K_mkt = df_slice[K_col].to_numpy(dtype=float)
    iv_mkt = df_slice[iv_col].to_numpy(dtype=float)

    k_mkt = np.log(K_mkt / float(F))
    w_mkt = (iv_mkt ** 2) * T
    
    # Grid for arbitrage checks
    k_grid = np.log(K_grid_cons / float(F)) if K_grid_cons is not None else np.linspace(-3, 3, 200)

    def obj(x):
        a, b, rho, m, sig = x
        
        # 1. Market Fit (MSE)
        w_model_mkt = svi_raw_total_variance(k_mkt, a, b, rho, m, sig)
        mse = np.mean((w_model_mkt - w_mkt)**2)
        
        # 2. Butterfly Penalty (Durrleman)
        w_g = svi_raw_total_variance(k_grid, a, b, rho, m, sig)
        w_p = svi_raw_first_derivative(k_grid, a, b, rho, m, sig)
        w_pp = svi_raw_second_derivative(k_grid, a, b, rho, m, sig)
        
        w_safe = np.maximum(w_g, 1e-8)
        g_k = (1.0 - (k_grid * w_p) / (2.0 * w_safe))**2 - (w_p**2 / 4.0) * (1.0 / w_safe + 0.25) + w_pp / 2.0
        
        # If g_k < 0, we penalize. 
        # Multiplier (1e4 to 1e6) depends on how strict you want to be.
        butterfly_penalty = 1e5 * np.sum(np.square(np.minimum(g_k, 0.0)))
        
        # 3. Calendar Penalty
        calendar_penalty = 0.0
        if prev_params is not None and F_prev is not None:
            k_prev = np.log(K_grid_cons / float(F_prev))
            w_prev = svi_raw_total_variance(k_prev, prev_params["a"], prev_params["b"], 
                                            prev_params["rho"], prev_params["m"], prev_params["sigma"])
            w_curr = svi_raw_total_variance(k_grid, a, b, rho, m, sig)
            
            # Penalize if current < previous
            calendar_penalty = 1e5 * np.sum(np.square(np.minimum(w_curr - w_prev, 0.0)))

        return mse + butterfly_penalty + calendar_penalty

    # Simple bounds to keep parameters in a physical range
    bounds = [(1e-6, None), (1e-2, 1.0), (-0.95, 0.95), (None, None), (1e-3, 0.5)]
    
    # Starting point
    if prev_params is not None:
        x0 = np.array([prev_params["a"], prev_params["b"], prev_params["rho"], prev_params["m"], prev_params["sigma"]])
    else:
        x0 = np.array([np.mean(w_mkt), 0.1, 0.0, 0.0, 0.1])

    # With soft penalties, you can often use 'BFGS' or 'L-BFGS-B', 
    # but 'SLSQP' still works fine with bounds.
    res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
    
    return {"a": res.x[0], "b": res.x[1], "rho": res.x[2], "m": res.x[3], "sigma": res.x[4], 
            "T": T, "F": float(F), "result": res}

def fit_surface_svi(df_iv, S0, rate, K_min, K_max):
    """Loops through maturities sorted in time, ensuring no total variance crossing."""
    maturities = sorted(df_iv["time_to_maturity"].unique())
    K_grid_cons = np.linspace(K_min, K_max, 100)
    
    surface_params = {}
    prev_params = None
    F_prev = None
    
    for T in maturities:
        df_slice = df_iv[df_iv["time_to_maturity"] == T]
        F = S0 * np.exp(rate * T)
        
        params = fit_svi_raw_one_slice(df_slice, F, prev_params=prev_params, F_prev=F_prev, K_grid_cons=K_grid_cons)
        surface_params[T] = params
        prev_params = params
        F_prev = F
        
    return surface_params

def extract_rnd_for_maturity_constrained(surface_params, T, S0, rate, n_points=3000):
    """Extracts RND solely using the calendar-safe SVI parameters. Avoids hard IV clamping to preserve limits."""
    K_min = max(S0 * 0.5, 1e-4)
    K_max = S0 * 1.5          
    K_grid = np.linspace(K_min, K_max, n_points)
    
    # 1. Generate IV directly from pre-fitted params
    params = surface_params[T]
    iv_grid = svi_iv_from_params(K_grid, params)
    
    # 2. Calculate BS Prices
    C_prices = np.array([black_scholes_price(S0, k, T, rate, sig) for k, sig in zip(K_grid, iv_grid)])
    dK = K_grid[1] - K_grid[0]
    
    # 3. Breeden-Litzenberger (Second derivative)
    second_derivative = (C_prices[2:] - 2*C_prices[1:-1] + C_prices[:-2]) / (dK**2)
    density = np.exp(rate * T) * second_derivative
    
    K_density = K_grid[1:-1]
    prob_mass = density * dK
    
    # Clean up and normalize
    prob_mass = np.maximum(prob_mass, 1e-12)
    prob_mass /= np.sum(prob_mass)
    
    return K_density, prob_mass