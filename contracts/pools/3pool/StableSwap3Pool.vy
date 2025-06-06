# @version 0.2.4
# (c) Curve.Fi, 2020
# Pool for DAI/USDC/USDT

from vyper.interfaces import ERC20

interface CurveToken:
    def totalSupply() -> uint256: view
    def mint(_to: address, _value: uint256) -> bool: nonpayable
    def burnFrom(_to: address, _value: uint256) -> bool: nonpayable


# Events
event TokenExchange:
    buyer: indexed(address)
    sold_id: int128
    tokens_sold: uint256
    bought_id: int128
    tokens_bought: uint256


event AddLiquidity:
    provider: indexed(address)
    token_amounts: uint256[N_COINS]
    fees: uint256[N_COINS]
    invariant: uint256
    token_supply: uint256

event RemoveLiquidity:
    provider: indexed(address)
    token_amounts: uint256[N_COINS]
    fees: uint256[N_COINS]
    token_supply: uint256

event RemoveLiquidityOne:
    provider: indexed(address)
    token_amount: uint256
    coin_amount: uint256

event RemoveLiquidityImbalance:
    provider: indexed(address)
    token_amounts: uint256[N_COINS]
    fees: uint256[N_COINS]
    invariant: uint256
    token_supply: uint256

event CommitNewAdmin:
    deadline: indexed(uint256)
    admin: indexed(address)

event NewAdmin:
    admin: indexed(address)


event CommitNewFee:
    deadline: indexed(uint256)
    fee: uint256
    admin_fee: uint256

event NewFee:
    fee: uint256
    admin_fee: uint256

event RampA:
    old_A: uint256
    new_A: uint256
    initial_time: uint256
    future_time: uint256

event StopRampA:
    A: uint256
    t: uint256


# This can (and needs to) be changed at compile time
N_COINS: constant(int128) = 3  # <- change

FEE_DENOMINATOR: constant(uint256) = 10 ** 10
LENDING_PRECISION: constant(uint256) = 10 ** 18
PRECISION: constant(uint256) = 10 ** 18  # The precision to convert to
PRECISION_MUL: constant(uint256[N_COINS]) = [1, 1000000000000, 1000000000000]
RATES: constant(uint256[N_COINS]) = [1000000000000000000, 1000000000000000000000000000000, 1000000000000000000000000000000]
FEE_INDEX: constant(int128) = 2  # Which coin may potentially have fees (USDT)

MAX_ADMIN_FEE: constant(uint256) = 10 * 10 ** 9
MAX_FEE: constant(uint256) = 5 * 10 ** 9
MAX_A: constant(uint256) = 10 ** 6
MAX_A_CHANGE: constant(uint256) = 10

ADMIN_ACTIONS_DELAY: constant(uint256) = 3 * 86400
MIN_RAMP_TIME: constant(uint256) = 86400

coins: public(address[N_COINS])
balances: public(uint256[N_COINS]) # stores the balances of the coins in the pool
fee: public(uint256)  # fee * 1e10
admin_fee: public(uint256)  # admin_fee * 1e10

owner: public(address)
token: CurveToken

initial_A: public(uint256)
future_A: public(uint256)
initial_A_time: public(uint256)
future_A_time: public(uint256)

admin_actions_deadline: public(uint256)
transfer_ownership_deadline: public(uint256)
future_fee: public(uint256)
future_admin_fee: public(uint256)
future_owner: public(address)

is_killed: bool
kill_deadline: uint256
KILL_DEADLINE_DT: constant(uint256) = 2 * 30 * 86400


@external
def __init__(
    _owner: address,
    _coins: address[N_COINS],
    _pool_token: address,
    _A: uint256,
    _fee: uint256,
    _admin_fee: uint256
):
    """
    @notice Contract constructor - Initializes a new StableSwap pool
    @dev This constructor sets up a Curve Finance StableSwap pool for 3 stablecoins
    
    Pool Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    StableSwap 3Pool                             │
    ├─────────────────────────────────────────────────────────────────┤
    │  Token 0: DAI    (18 decimals)                                 │
    │  Token 1: USDC   (6 decimals → normalized to 18)               │
    │  Token 2: USDT   (6 decimals → normalized to 18)               │
    ├─────────────────────────────────────────────────────────────────┤
    │  Amplification: A = _A                                          │
    │  Trading Fee: _fee / 1e10 (e.g., 4e6 = 0.04%)                 │
    │  Admin Fee: _admin_fee / 1e10 (e.g., 5e9 = 50% of trading fee) │
    └─────────────────────────────────────────────────────────────────┘
    
    Mathematical Foundation:
    The StableSwap invariant equation:
    An³ΣxᵢXᵢ + D = ADn³ + D⁴/(4ΠXᵢ)
    
    Where:
    - A = amplification coefficient (determines curve shape)
    - n = number of coins (3 in this case)
    - xᵢ = normalized balance of coin i
    - D = invariant (total virtual balance)
    
    Example Parameter Setup:
    - _A = 2000 (creates a relatively flat curve for stablecoins)
    - _fee = 4000000 (0.04% trading fee)
    - _admin_fee = 5000000000 (50% of trading fees go to admin)
    
    @param _owner Contract owner address (governance/DAO)
    @param _coins Array of 3 stablecoin addresses [DAI, USDC, USDT]
    @param _pool_token Address of the LP token contract (3CRV)
    @param _A Amplification coefficient (typically 100-5000 for stablecoins)
    @param _fee Trading fee in basis points * 1e6 (e.g., 4e6 = 0.04%)
    @param _admin_fee Admin fee as percentage of trading fee * 1e8
    """
    # Validate all coin addresses are non-zero to prevent deployment errors
    for i in range(N_COINS):
        assert _coins[i] != ZERO_ADDRESS, "Coin address cannot be zero"
    
    # Store the three stablecoin addresses
    self.coins = _coins
    
    # Initialize amplification coefficient for both current and future values
    # This allows for smooth A parameter changes via ramping
    self.initial_A = _A
    self.future_A = _A
    
    # Set fee structure (measured in 1e10 precision)
    self.fee = _fee          # Trading fee charged on swaps
    self.admin_fee = _admin_fee  # Percentage of trading fees for protocol
    
    # Set contract owner (typically a DAO or multisig)
    self.owner = _owner
    
    # Set emergency kill deadline (safety mechanism)
    self.kill_deadline = block.timestamp + KILL_DEADLINE_DT
    
    # Initialize LP token interface
    self.token = CurveToken(_pool_token)


@view
@internal
def _A() -> uint256:
    """
    @notice Calculate the current value of the amplification coefficient
    @dev Handles the linear interpolation between initial_A and future_A
    
    Linear interpolation formula: 
        currentA = initialA + (futureA - initialA) * timeProgress
    
    Where timeProgress = (currentTime - startTime) / (endTime - startTime)
    
    Visualization of A changing over time:
    
    Amplification (A)
    │
    │                                   (future_time, future_A)
    │                                 •
    │                              •
    │                           •
    │                        •
    │                     •
    │                  •
    │               •
    │            •
    │         •
    │      •
    │   •
    │ •
    │• (initial_time, initial_A)
    │
    └───────────────────────────────────── Time
    
    This ensures a smooth and predictable change in the pool's behavior.
    
    @return Current value of A
    """
    t1: uint256 = self.future_A_time  # End time of the ramp
    A1: uint256 = self.future_A       # Target value of A

    if block.timestamp < t1:
        # We're in the middle of a ramp, need to calculate the current A
        A0: uint256 = self.initial_A  # Starting value of A
        t0: uint256 = self.initial_A_time  # Start time of the ramp
        
        # Calculate the progress of the ramp (0 to 1)
        # time_progress = (current_time - start_time) / (end_time - start_time)
        time_progress: uint256 = (block.timestamp - t0) / (t1 - t0)
        
        # Vyper doesn't allow direct handling of negative numbers in uint256,
        # so we handle increasing and decreasing A separately
        if A1 > A0:
            # A is increasing
            # Example: If A0=100, A1=200, and we're halfway through the ramp:
            # A = 100 + (200-100) * 0.5 = 100 + 50 = 150
            return A0 + (A1 - A0) * time_progress
        else:
            # A is decreasing
            # Example: If A0=200, A1=100, and we're halfway through the ramp:
            # A = 200 - (200-100) * 0.5 = 200 - 50 = 150
            return A0 - (A0 - A1) * time_progress
    else:
        # Ramp has completed, or no ramp is in progress
        return A1

    else:  # when t1 == 0 or block.timestamp >= t1
        return A1
 

@view
@external
def A() -> uint256:
    """
    @notice Get the current amplification coefficient
    @dev Public getter for the current A value, accounting for any ongoing ramps
    
    This function provides external access to the pool's amplification coefficient,
    which is crucial for understanding the current behavior of the pool.
    
    Usage Examples:
    - Frontend interfaces display current A value to users
    - Arbitrage bots check A to calculate optimal trade sizes
    - Analytics platforms track A changes over time
    
    Mathematical Impact of A on Trading:
    
    High A (e.g., A=5000):                Low A (e.g., A=50):
    ┌─────────────────────┐               ┌─────────────────────┐
    │     Almost Flat     │               │     Curved Like     │
    │    ____________     │               │        ____         │
    │   /            \    │               │       /    \        │
    │  /              \   │               │      /      \       │
    │ /                \  │               │     /        \      │
    │/                  \ │               │    /          \     │
    └─────────────────────┘               └─────────────────────┘
       Minimal slippage                      Higher slippage
       Perfect for 1:1 swaps                More like Uniswap
    
    @return Current amplification coefficient
    """
    return self._A()


@view
@internal
def _xp() -> uint256[N_COINS]:
    """
    @notice Normalize token balances to 18 decimal precision for calculations
    @dev This function is critical for ensuring mathematical consistency across tokens
         with different decimal places
    
    Precision Normalization Process:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Raw Token Balances                          │
    ├─────────────────────────────────────────────────────────────────┤
    │  DAI:  1000.5 * 10^18  (18 decimals - native)                 │
    │  USDC: 1000.5 * 10^6   (6 decimals - needs scaling)           │
    │  USDT: 1000.5 * 10^6   (6 decimals - needs scaling)           │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │                 Normalized Balances (xp)                       │
    ├─────────────────────────────────────────────────────────────────┤
    │  DAI:  1000.5 * 10^18  (18 decimals)                          │
    │  USDC: 1000.5 * 10^18  (scaled to 18 decimals)                │
    │  USDT: 1000.5 * 10^18  (scaled to 18 decimals)                │
    └─────────────────────────────────────────────────────────────────┘
    
    Mathematical Transformation:
    For each token i:
    xp[i] = RATES[i] * balance[i] / LENDING_PRECISION
    
    Where RATES are:
    - DAI:  1e18 (no scaling needed)
    - USDC: 1e30 (scales 6 decimals to 18: 1e30/1e18 = 1e12 multiplier)
    - USDT: 1e30 (scales 6 decimals to 18: 1e30/1e18 = 1e12 multiplier)
    
    Example Calculation for USDC:
    Raw USDC balance: 1,000,500,000 (1000.5 USDC in 6 decimals)
    xp[1] = 1e30 * 1,000,500,000 / 1e18
          = 1,000,500,000 * 1e12
          = 1,000,500,000,000,000,000,000 (1000.5 in 18 decimals)
    
    @return Array of normalized token balances in 18 decimal precision
    """
    result: uint256[N_COINS] = RATES
    for i in range(N_COINS):
        # Normalize each token balance to 18 decimal precision
        # This ensures all mathematical operations work consistently
        # regardless of the underlying token's decimal places
        result[i] = result[i] * self.balances[i] / LENDING_PRECISION
    return result


@pure
@internal
def _xp_mem(_balances: uint256[N_COINS]) -> uint256[N_COINS]:
    """
    @notice Normalize provided token balances to 18 decimal precision
    @dev Pure function version of _xp() that works with provided balances
         instead of stored balances. Used for simulating calculations.
    
    Key Difference from _xp():
    ┌─────────────────┬─────────────────┬─────────────────────────┐
    │    Function     │   Data Source   │      Use Case           │
    ├─────────────────┼─────────────────┼─────────────────────────┤
    │ _xp()          │ self.balances   │ Current pool state      │
    │ _xp_mem()      │ _balances param │ Simulated calculations  │
    └─────────────────┴─────────────────┴─────────────────────────┘
    
    Usage Scenarios:
    1. Calculating swap outputs before execution
    2. Estimating LP tokens for deposits
    3. Simulating withdrawals
    4. Fee calculations
    
    Mathematical Process:
    For each token i in the provided balances:
    normalized[i] = RATES[i] * _balances[i] / PRECISION
    
    Example with hypothetical balances:
    Input: [1000e18, 1000e6, 1000e6] (DAI, USDC, USDT)
    
    DAI:  1e18 * 1000e18 / 1e18 = 1000e18
    USDC: 1e30 * 1000e6 / 1e18 = 1000e18
    USDT: 1e30 * 1000e6 / 1e18 = 1000e18
    
    Output: [1000e18, 1000e18, 1000e18] (all normalized to 18 decimals)
    
    @param _balances Array of token balances to normalize
    @return Array of normalized token balances in 18 decimal precision
    """
    result: uint256[N_COINS] = RATES
    for i in range(N_COINS):
        # Apply the same normalization as _xp() but to provided balances
        # This allows for calculations without modifying pool state
        result[i] = result[i] * _balances[i] / PRECISION
    return result


@pure
@internal
def get_D(xp: uint256[N_COINS], amp: uint256) -> uint256:
    """
    @notice Calculate the StableSwap invariant D using Newton's method
    @dev This is the heart of the StableSwap algorithm - calculates the invariant
         that represents the total virtual balance of the pool
    
    Mathematical Foundation - StableSwap Invariant:
    An³ΣxᵢXᵢ + D = ADn³ + D⁴/(4ΠXᵢ)
    
    Rearranging to solve for D:
    D⁴ + (An³ - 1)D³S + An³D²S - An³DS² = 0
    
    Where:
    - A = amplification coefficient
    - n = number of coins (3)
    - xᵢ = balance of coin i (normalized)
    - D = invariant (what we're solving for)
    - S = Σxᵢ = sum of all balances
    
    Newton's Method Visualization:
    
        f(D)
         │
         │     f'(D₁) slope
         │       ╱
         │      ╱
         │     ╱
         │    ╱    f(D₁)
         │   ╱      •
         │  ╱      ╱│
         │ ╱      ╱ │
         │╱      ╱  │
    ─────┼──────•───┼─────────► D
         │     D₂   D₁
         │
    D₂ = D₁ - f(D₁)/f'(D₁)
    
    Each iteration gets us closer to the root where f(D) = 0
    
    Practical Example:
    If we have balanced pool with:
    - DAI: 1000e18
    - USDC: 1000e18 (normalized)  
    - USDT: 1000e18 (normalized)
    - A = 2000
    
    Then D ≈ 3000e18 (representing total virtual balance)
    
    Convergence Properties:
    - Typically converges in 3-5 iterations
    - Precision: within 1 wei
    - Guaranteed convergence for valid inputs
    
    @param xp Array of normalized token balances
    @param amp Amplification coefficient
    @return D The calculated invariant representing total virtual balance
    """
    # S = sum of all normalized balances
    S: uint256 = 0
    for _x in xp:
        S += _x
    
    # If pool is empty, invariant is zero
    if S == 0:
        return 0

    # Initialize Newton's method
    Dprev: uint256 = 0      # Previous iteration value
    D: uint256 = S          # Initial guess: D = S (sum of balances)
    Ann: uint256 = amp * N_COINS  # A * n for efficiency

    # Newton's method iteration (maximum 255 iterations for safety)
    for _i in range(255):
        # Calculate D_P = D^(n+1) / (n^n * ∏xᵢ)
        # This represents the derivative term in Newton's method
        D_P: uint256 = D
        for _x in xp:
            # D_P = D_P * D / (_x * N_COINS)
            # If any balance is 0, this will cause revert - only withdrawal will work
            # This is intentional safety behavior
            D_P = D_P * D / (_x * N_COINS)
        
        # Store previous D for convergence check
        Dprev = D
        
        # Newton's method update formula:
        # D_new = (Ann*S + D_P*n) * D / ((Ann-1)*D + (n+1)*D_P)
        # This comes from the derivative of the StableSwap invariant equation
        D = (Ann * S + D_P * N_COINS) * D / ((Ann - 1) * D + (N_COINS + 1) * D_P)
        
        # Check for convergence (precision of 1 wei)
        if D > Dprev:
            if D - Dprev <= 1:
                break  # Converged from below
        else:
            if Dprev - D <= 1:
                break  # Converged from above
    
    return D


@view
@internal
def get_D_mem(_balances: uint256[N_COINS], amp: uint256) -> uint256:
    """
    @notice Calculate the StableSwap invariant D for provided balances
    @dev Wrapper function that normalizes balances and calculates D without
         modifying pool state. Essential for simulating operations.
    
    Function Flow:
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │   Raw Balances  │───▶│   _xp_mem()     │───▶│    get_D()      │
    │                 │    │   (normalize)   │    │  (Newton's)     │
    │ [1000e18,       │    │                 │    │                 │
    │  1000e6,        │    │ [1000e18,       │    │                 │
    │  1000e6]        │    │  1000e18,       │    │   D = 3000e18   │
    └─────────────────┘    │  1000e18]       │    └─────────────────┘
                           └─────────────────┘
    
    Use Cases:
    1. Swap Calculations: Calculate D before and after swap to determine fees
    2. Liquidity Estimation: Estimate LP tokens for deposit amounts
    3. Withdrawal Planning: Calculate balanced withdrawal amounts
    4. Price Impact Analysis: Compare D values for large trades
    
    Example Usage in Swap:
    ```
    # Before swap: DAI=1000, USDC=1000, USDT=1000
    D_before = get_D_mem([1000e18, 1000e6, 1000e6], A) # ≈ 3000e18
    
    # After swap: DAI=900, USDC=1100, USDT=1000 (theoretical)
    D_after = get_D_mem([900e18, 1100e6, 1000e6], A)   # ≈ 3000e18
    
    # D should remain constant (minus fees)
    ```
    
    Mathematical Relationship:
    D represents the total "virtual balance" of the pool. For a balanced pool:
    D ≈ n * average_balance_normalized
    
    Where n is the number of coins (3 in this case).
    
    @param _balances Array of raw token balances to analyze
    @param amp Amplification coefficient to use in calculation
    @return D The calculated invariant for the provided balances
    """
    return self.get_D(self._xp_mem(_balances), amp)


@view
@external
def get_virtual_price() -> uint256:
    """
    @notice Calculate the virtual price of LP tokens (portfolio value per token)
    @dev This represents the value of each LP token in terms of the underlying assets,
         scaled up by 1e18. Used for calculating profit and portfolio performance.
    
    Mathematical Formula:
    virtual_price = D * PRECISION / total_supply
    
    Where:
    - D = StableSwap invariant (total virtual balance)
    - PRECISION = 1e18 (for decimal scaling)
    - total_supply = total LP tokens in circulation
    
    Economic Interpretation:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Virtual Price Analysis                       │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Virtual Price = Pool Value / LP Token Supply                  │
    │                                                                 │
    │  • At launch: virtual_price ≈ 1.0 (1e18)                      │
    │  • Over time: virtual_price > 1.0 (due to trading fees)       │
    │  • Growth rate: indicator of pool profitability                │
    │                                                                 │
    │  Example Timeline:                                              │
    │  Day 0:  virtual_price = 1.000000 (pool launch)               │
    │  Day 30: virtual_price = 1.002341 (fees accumulated)          │
    │  Day 365: virtual_price = 1.028492 (2.8% annual return)       │
    └─────────────────────────────────────────────────────────────────┘
    
    Practical Example:
    Pool State:
    - DAI: 1,000,000 tokens (1e24 wei)
    - USDC: 1,000,000 tokens (1e12 wei, normalized to 1e24)
    - USDT: 1,000,000 tokens (1e12 wei, normalized to 1e24)
    - Total D ≈ 3,000,000e18
    - LP tokens minted: 2,950,000e18 (slightly less due to fees)
    
    virtual_price = 3,000,000e18 * 1e18 / 2,950,000e18
                  = 1.016949152542372881e18
                  ≈ 1.017 (1.7% premium due to accumulated fees)
    
    Use Cases:
    1. Portfolio Tracking: Calculate USD value of LP positions
    2. Yield Farming: Measure LP token appreciation over time
    3. Arbitrage: Compare virtual price across different pools
    4. Analytics: Track pool performance and fee accumulation
    
    Important Notes:
    - Virtual price should only increase over time (due to fees)
    - Sudden decreases might indicate bugs or exploits
    - Used by many protocols for LP token valuation
    
    @return Virtual price of each LP token, scaled by 1e18
    """
    # Calculate current pool invariant using actual balances and A parameter
    D: uint256 = self.get_D(self._xp(), self._A())
    
    # Get total supply of LP tokens currently in circulation
    token_supply: uint256 = self.token.totalSupply()
    
    # Calculate value per LP token
    # D is in normalized precision (18 decimals for all tokens)
    # Result represents the underlying asset value per LP token
    return D * PRECISION / token_supply


@view
@external
def calc_token_amount(amounts: uint256[N_COINS], deposit: bool) -> uint256:
    """
    @notice Calculate LP token amount for deposit/withdrawal without fees
    @dev Simplified calculation that excludes fees - used for slippage estimation
         and front-running protection. NOT for precise fee calculations!
    
    Mathematical Process:
    ┌─────────────────────────────────────────────────────────────────┐
    │               Token Amount Calculation Flow                     │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Get current pool state: D₀ = f(current_balances)           │
    │  2. Simulate operation: D₁ = f(new_balances)                   │
    │  3. Calculate change ratio: (D₁ - D₀) / D₀                     │
    │  4. Apply to LP supply: change * total_supply / D₀             │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Deposit Example:
    Current state: [1000 DAI, 1000 USDC, 1000 USDT]
    D₀ = 3000 (approximately)
    LP supply = 2950
    
    User deposits: [100 DAI, 0 USDC, 0 USDT]
    New state: [1100 DAI, 1000 USDC, 1000 USDT]  
    D₁ = 3095 (approximately)
    
    LP tokens = (3095 - 3000) * 2950 / 3000 = 93.42 LP tokens
    
    Withdrawal Example:
    User withdraws: [50 DAI, 50 USDC, 50 USDT]
    New state: [950 DAI, 950 USDC, 950 USDT]
    D₁ = 2850 (approximately)
    
    LP tokens = (3000 - 2850) * 2950 / 3000 = 147.5 LP tokens to burn
    
    Key Properties:
    1. Proportional: Equal value deposits yield similar LP amounts
    2. Slippage-aware: Imbalanced deposits yield fewer LP tokens
    3. Fee-free: This calculation excludes trading fees
    4. Approximate: Real calculations include fees and bonuses
    
    Use Cases:
    - Frontend: Display estimated LP tokens before transaction
    - MEV Protection: Prevent sandwich attacks with reasonable estimates
    - Slippage Calculation: Compare with actual amounts to detect high fees
    - User Interface: Show impact of deposit/withdrawal sizes
    
    Visual Representation for Imbalanced Deposit:
    
    Balanced Deposit (gets full value):         Imbalanced Deposit (gets less):
    ┌─────────────────────────────┐           ┌─────────────────────────────┐
    │  Before: [1000, 1000, 1000] │           │  Before: [1000, 1000, 1000] │
    │  Deposit: [100, 100, 100]   │           │  Deposit: [300, 0, 0]       │
    │  After: [1100, 1100, 1100]  │           │  After: [1300, 1000, 1000]  │
    │                             │           │                             │
    │  LP tokens: ~296 (full)     │           │  LP tokens: ~285 (penalty)  │
    │  Efficiency: 100%           │           │  Efficiency: ~96%           │
    └─────────────────────────────┘           └─────────────────────────────┘
    
    @param amounts Array of token amounts to deposit/withdraw
    @param deposit True for deposits, False for withdrawals
    @return Estimated LP token amount (positive for mint, burn amount for withdrawal)
    """
    # Get current pool balances
    _balances: uint256[N_COINS] = self.balances
    amp: uint256 = self._A()
    
    # Calculate current invariant D₀
    D0: uint256 = self.get_D_mem(_balances, amp)
    
    # Simulate the operation by updating balances
    for i in range(N_COINS):
        if deposit:
            _balances[i] += amounts[i]  # Add deposited amounts
        else:
            _balances[i] -= amounts[i]  # Subtract withdrawn amounts
    
    # Calculate new invariant D₁ with updated balances
    D1: uint256 = self.get_D_mem(_balances, amp)
    
    # Get current LP token supply
    token_amount: uint256 = self.token.totalSupply()
    
    # Calculate the change in invariant
    diff: uint256 = 0
    if deposit:
        diff = D1 - D0  # Invariant increase
    else:
        diff = D0 - D1  # Invariant decrease
    
    # Calculate LP token amount proportional to invariant change
    # LP_tokens = (ΔD / D₀) * total_supply
    return diff * token_amount / D0

@external
@nonreentrant('lock')
def add_liquidity(amounts: uint256[N_COINS], min_mint_amount: uint256):
    """
    @notice Add liquidity to the pool and receive LP tokens
    @dev This function handles both balanced and imbalanced deposits, applying
         fees for imbalanced deposits to prevent arbitrage exploitation
    
    Mathematical Process Overview:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Add Liquidity Flow                           │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Calculate D₀ (current invariant)                           │
    │  2. Transfer tokens from user                                   │
    │  3. Calculate D₁ (invariant after deposit)                     │
    │  4. Calculate imbalance fees                                    │
    │  5. Calculate D₂ (invariant after fees)                        │
    │  6. Mint LP tokens proportional to (D₂ - D₀) / D₀              │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Fee Calculation for Imbalanced Deposits:
    
    Balanced Deposit (no fees):           Imbalanced Deposit (with fees):
    ┌─────────────────────────────┐       ┌─────────────────────────────┐
    │  Before: [1000, 1000, 1000] │       │  Before: [1000, 1000, 1000] │
    │  Deposit: [100, 100, 100]   │       │  Deposit: [300, 0, 0]       │
    │  Ideal: [1100, 1100, 1100]  │       │  Ideal: [1100, 1100, 1100]  │
    │  Actual: [1100, 1100, 1100] │       │  Actual: [1300, 1000, 1000] │
    │                             │       │                             │
    │  Imbalance: [0, 0, 0]       │       │  Imbalance: [200, -100, -100]│
    │  Fees: [0, 0, 0]            │       │  Fees: [fee*200, fee*100, fee*100]│
    └─────────────────────────────┘       └─────────────────────────────┘
    
    Fee Formula:
    fee_per_coin = base_fee * |actual_balance - ideal_balance| / FEE_DENOMINATOR
    
    Where:
    - base_fee = pool_fee * N_COINS / (4 * (N_COINS - 1))
    - ideal_balance = new_D * old_balance / old_D
    - FEE_DENOMINATOR = 10^10
    
    Example Calculation:
    Pool state: [1000, 1000, 1000] DAI/USDC/USDT
    Deposit: [300, 0, 0] DAI
    Pool fee: 0.04% = 4e6
    
    1. D₀ = 3000 (current invariant)
    2. D₁ = 3296 (after deposit, before fees)
    3. Ideal balances: [1099, 1099, 1099]
    4. Actual balances: [1300, 1000, 1000]
    5. Imbalances: [201, -99, -99]
    6. Fees per coin: [0.67, 0.33, 0.33] tokens
    7. D₂ = 3294 (after fees)
    8. LP tokens = total_supply * (3294 - 3000) / 3000
    
    Security Features:
    - Reentrancy protection via @nonreentrant
    - Pool kill switch check
    - Minimum mint amount protection (slippage)
    - Safe token transfers with return value checks
    - Special handling for fee-on-transfer tokens (USDT)
    
    @param amounts Array of token amounts to deposit [DAI, USDC, USDT]
    @param min_mint_amount Minimum LP tokens to receive (slippage protection)
    """
    # Ensure pool is not in emergency shutdown mode
    assert not self.is_killed  # dev: is killed
    
    # Initialize fee tracking array for imbalanced deposits
    fees: uint256[N_COINS] = empty(uint256[N_COINS])
    
    # Calculate reduced fee for imbalanced liquidity provision
    # fee = pool_fee * N_COINS / (4 * (N_COINS - 1))
    # This is 1/4 of the swap fee, making imbalanced deposits less penalized
    _fee: uint256 = self.fee * N_COINS / (4 * (N_COINS - 1))
    _admin_fee: uint256 = self.admin_fee
    amp: uint256 = self._A()

    # Get current LP token supply and pool state
    token_supply: uint256 = self.token.totalSupply()
    D0: uint256 = 0  # Current invariant
    old_balances: uint256[N_COINS] = self.balances
    
    # Calculate current invariant (if pool has liquidity)
    if token_supply > 0:
        D0 = self.get_D_mem(old_balances, amp)
    
    # Working copy of balances for calculations
    new_balances: uint256[N_COINS] = old_balances

    # Process each token deposit
    for i in range(N_COINS):
        in_amount: uint256 = amounts[i]
        
        # First deposit must include all coins to prevent attacks
        if token_supply == 0:
            assert in_amount > 0  # dev: initial deposit requires all coins
        
        in_coin: address = self.coins[i]

        # Transfer tokens from user to pool
        if in_amount > 0:
            # Special handling for fee-on-transfer tokens (USDT)
            if i == FEE_INDEX:
                in_amount = ERC20(in_coin).balanceOf(self)

            # Safe transfer from user - handles tokens that don't return bool
            _response: Bytes[32] = raw_call(
                in_coin,
                concat(
                    method_id("transferFrom(address,address,uint256)"),
                    convert(msg.sender, bytes32),
                    convert(self, bytes32),
                    convert(amounts[i], bytes32),
                ),
                max_outsize=32,
            )  # dev: failed transfer
            
            # Check transfer success for tokens that return bool
            if len(_response) > 0:
                assert convert(_response, bool)  # dev: failed transfer
            
            # Calculate actual received amount for fee-on-transfer tokens
            if i == FEE_INDEX:
                in_amount = ERC20(in_coin).balanceOf(self) - in_amount

        # Update balance with actual received amount
        new_balances[i] = old_balances[i] + in_amount

    # Calculate invariant after deposits (before fees)
    D1: uint256 = self.get_D_mem(new_balances, amp)
    assert D1 > D0  # Invariant must increase

    # Calculate imbalance fees and update balances
    D2: uint256 = D1  # Invariant after fees
    
    if token_supply > 0:
        # Apply imbalance fees (not applicable for first deposit)
        for i in range(N_COINS):
            # Calculate ideal balance: what this coin's balance should be
            # if the deposit was perfectly balanced
            ideal_balance: uint256 = D1 * old_balances[i] / D0
            
            # Calculate deviation from ideal balance
            difference: uint256 = 0
            if ideal_balance > new_balances[i]:
                difference = ideal_balance - new_balances[i]  # Under-deposited
            else:
                difference = new_balances[i] - ideal_balance  # Over-deposited
            
            # Calculate imbalance fee for this token
            fees[i] = _fee * difference / FEE_DENOMINATOR
            
            # Update pool balance: new_balance - admin_fee_portion
            self.balances[i] = new_balances[i] - (fees[i] * _admin_fee / FEE_DENOMINATOR)
            
            # Subtract total fee from calculation balance
            new_balances[i] -= fees[i]
        
        # Recalculate invariant after fee deduction
        D2 = self.get_D_mem(new_balances, amp)
    else:
        # First deposit: no fees, direct balance update
        self.balances = new_balances

    # Calculate LP tokens to mint
    mint_amount: uint256 = 0
    if token_supply == 0:
        # First deposit: mint amount equal to invariant
        mint_amount = D1  # Include any rounding dust
    else:
        # Subsequent deposits: mint proportional to invariant increase
        # LP_tokens = total_supply * (D₂ - D₀) / D₀
        mint_amount = token_supply * (D2 - D0) / D0

    # Slippage protection: ensure user gets minimum expected LP tokens
    assert mint_amount >= min_mint_amount, "Slippage screwed you"

    # Mint LP tokens to user
    self.token.mint(msg.sender, mint_amount)

    # Emit event for tracking
    log AddLiquidity(msg.sender, amounts, fees, D1, token_supply + mint_amount)


@view
@internal
def get_y(i: int128, j: int128, x: uint256, xp_: uint256[N_COINS]) -> uint256:
    """
    @notice Calculate the new balance of token j after swapping token i
    @dev Uses Newton's method to solve for the new balance that maintains the invariant
         This is the core calculation behind all swaps in the StableSwap pool
    
    Mathematical Foundation:
    We need to solve for y (new balance of token j) in the StableSwap invariant:
    An³ΣxᵢXᵢ + D = ADn³ + D⁴/(4ΠXᵢ)
    
    Rearranging for y when we know x (new balance of token i):
    y² + y(b - D) = c
    
    Where:
    - b = S + D/A  (S = sum of other token balances)
    - c = D³/(An³ΠXₖ) for k ≠ j
    - A = amplification coefficient
    - D = invariant (constant)
    
    Newton's Method for Solving y² + y(b - D) - c = 0:
    
    f(y) = y² + y(b - D) - c
    f'(y) = 2y + (b - D)
    
    y_new = y_old - f(y_old)/f'(y_old)
          = y_old - (y_old² + y_old(b-D) - c)/(2y_old + b - D)
          = (y_old² + c)/(2y_old + b - D)
    
    Convergence Visualization:
    
         y
         │
         │     Quadratic curve: f(y) = y² + y(b-D) - c
         │          ╱
         │        ╱
         │      ╱
         │    ╱
    ─────┼──╱─────────────► 
         │╱      y (balance)
         
    Each iteration moves closer to where f(y) = 0
    
    Example Swap Calculation:
    Pool: [1000 DAI, 1000 USDC, 1000 USDT]
    User swaps: 100 DAI → ? USDC
    
    1. Current balances: xp = [1000e18, 1000e18, 1000e18]
    2. New DAI balance: x = 1100e18 (1000 + 100)
    3. Solve for new USDC balance: y = ?
    4. Newton's method converges to: y ≈ 900.12e18
    5. USDC out = 1000e18 - 900.12e18 = 99.88 USDC
    
    Key Properties:
    - Maintains constant invariant D
    - Converges rapidly (typically 3-5 iterations)
    - Higher A = less slippage for balanced pools
    - Works for any token pair in the pool
    
    @param i Index of input token (being sold)
    @param j Index of output token (being bought)  
    @param x New balance of input token i after the trade
    @param xp_ Array of current normalized token balances
    @return y New balance of output token j after maintaining invariant
    """
    # Validate token indices to prevent invalid swaps
    assert i != j       # dev: same coin
    assert j >= 0       # dev: j below zero
    assert j < N_COINS  # dev: j above N_COINS
    assert i >= 0       # dev: i below zero  
    assert i < N_COINS  # dev: i above N_COINS

    # Get current amplification and invariant
    amp: uint256 = self._A()
    D: uint256 = self.get_D(xp_, amp)
    
    # Initialize variables for Newton's method
    c: uint256 = D          # Will become c = D³/(An³ΠXₖ) 
    S_: uint256 = 0         # Sum of balances excluding token j
    Ann: uint256 = amp * N_COINS  # A * n for efficiency

    # Build the equation coefficients
    # We need: c = D * D * D / (Ann * N_COINS * ∏Xₖ) for k ≠ j
    # And: S = ∑Xₖ for k ≠ j
    _x: uint256 = 0
    for _i in range(N_COINS):
        if _i == i:
            # Use the new balance for input token
            _x = x
        elif _i != j:
            # Use current balance for other tokens
            _x = xp_[_i]
        else:
            # Skip output token j - we're solving for its new balance
            continue
        
        S_ += _x  # Add to sum
        c = c * D / (_x * N_COINS)  # Build c coefficient
    
    # Complete the c calculation
    c = c * D / (Ann * N_COINS)
    
    # Calculate b coefficient: b = S + D/Ann
    b: uint256 = S_ + D / Ann
    
    # Newton's method iteration
    y_prev: uint256 = 0
    y: uint256 = D  # Initial guess: y = D
    
    for _i in range(255):  # Maximum 255 iterations for safety
        y_prev = y
        
        # Newton's method update: y = (y² + c) / (2y + b - D)
        # This comes from: y_new = y - f(y)/f'(y)
        # Where f(y) = y² + y(b-D) - c
        y = (y * y + c) / (2 * y + b - D)
        
        # Check convergence (precision of 1 wei)
        if y > y_prev:
            if y - y_prev <= 1:
                break  # Converged from below
        else:
            if y_prev - y <= 1:
                break  # Converged from above
    
    return y


@view
@external
def get_dy(i: int128, j: int128, dx: uint256) -> uint256:
    """
    @notice Calculate output amount for a given input (with fees)
    @dev This is the main pricing function for swaps, incorporating trading fees
         Used by frontends, aggregators, and arbitrage bots for price discovery
    
    Mathematical Process:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Swap Calculation Flow                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Convert input amount to normalized precision                │
    │  2. Calculate new input token balance: x_new = x_old + dx       │
    │  3. Solve StableSwap invariant for new output balance: y_new   │
    │  4. Calculate raw output: dy_raw = y_old - y_new               │
    │  5. Apply trading fee: dy_final = dy_raw - fee                 │
    │  6. Convert back to token's native precision                   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Precision Normalization:
    All calculations performed at 18-decimal precision for consistency
    
    Token Precision Handling:
    ┌─────────────────────────────────────────────────────────────────┐
    │  DAI:  18 decimals  →  rate = 1e18     →  no conversion        │
    │  USDC:  6 decimals  →  rate = 1e30     →  multiply by 1e12     │
    │  USDT:  6 decimals  →  rate = 1e30     →  multiply by 1e12     │
    └─────────────────────────────────────────────────────────────────┘
    
    Example Swap Calculation:
    Swap 100 USDC (6 decimals) → DAI (18 decimals)
    
    Step 1: Convert to normalized precision
    - dx = 100 * 1e6 = 100,000,000 (USDC units)
    - dx_normalized = 100,000,000 * 1e30 / 1e18 = 100e18
    
    Step 2: Calculate new USDC balance
    - Current USDC: 1000e18 (normalized)
    - New USDC: 1000e18 + 100e18 = 1100e18
    
    Step 3: Solve for new DAI balance using get_y()
    - Current DAI: 1000e18
    - New DAI: ~900.12e18 (maintains invariant)
    
    Step 4: Calculate raw output
    - dy_raw = 1000e18 - 900.12e18 = 99.88e18
    
    Step 5: Apply trading fee (0.04% = 4e6/1e10)
    - fee = 99.88 * 4e6 / 1e10 = 0.04 DAI
    - dy_final = 99.88 - 0.04 = 99.84 DAI
    
    Step 6: Convert to DAI precision (18 decimals)
    - Output = 99.84e18 (already 18 decimals)
    
    Slippage Analysis:
    Perfect exchange rate: 100 USDC = 100 DAI
    Actual output: 99.84 DAI
    Slippage: (100 - 99.84) / 100 = 0.16% = 16 basis points
    
    Visual Representation:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      StableSwap Curve                          │
    │   USDC                                                    DAI   │
    │    │                                                       │    │
    │    │          Current Point: (1000, 1000)                 │    │
    │    │              ●                                        │    │
    │    │            ╱   ╲                                      │    │
    │    │          ╱       ╲                                    │    │
    │    │        ╱           ╲                                  │    │
    │    │      ╱               ╲                                │    │
    │    │    ╱                   ╲                              │    │
    │    │  ╱        New Point:     ╲                            │    │
    │    │╱          (1100, 900.12)   ╲                          │    │
    │    └────────────────────────────────────────────────────────────┘
    │                       Invariant: D = constant                   │
    └─────────────────────────────────────────────────────────────────┘
    
    Use Cases:
    1. Frontend: Display swap quotes to users
    2. Aggregators: Compare prices across different pools
    3. Arbitrage: Identify profitable trading opportunities  
    4. MEV: Calculate sandwich attack profitability
    5. Risk Management: Assess slippage for large trades
    
    @param i Index of input token being sold
    @param j Index of output token being bought
    @param dx Amount of input token (in token's native precision)
    @return Amount of output token after fees (in token's native precision)
    """
    # Get rate multipliers and current normalized balances
    rates: uint256[N_COINS] = RATES
    xp: uint256[N_COINS] = self._xp()

    # Convert input amount to normalized precision and calculate new input balance
    x: uint256 = xp[i] + (dx * rates[i] / PRECISION)
    
    # Solve StableSwap invariant for new output token balance
    y: uint256 = self.get_y(i, j, x, xp)
    
    # Calculate raw output amount (subtracting 1 wei for precision safety)
    # Convert back to token's native precision
    dy: uint256 = (xp[j] - y - 1) * PRECISION / rates[j]
    
    # Calculate and subtract trading fee
    _fee: uint256 = self.fee * dy / FEE_DENOMINATOR
    return dy - _fee


@view
@external
def get_dy_underlying(i: int128, j: int128, dx: uint256) -> uint256:
    """
    @notice Calculate output amount for yield-bearing assets (underlying units)
    @dev Similar to get_dy but works with underlying token amounts for yield-bearing assets
         This function is essential for pools containing wrapped tokens (like cTokens, aTokens)
         where users want to know outputs in terms of underlying assets
    
    Key Differences from get_dy():
    ┌─────────────────────────────────────────────────────────────────┐
    │               get_dy() vs get_dy_underlying()                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  get_dy():          Uses rate multipliers (RATES)              │
    │                     For normal tokens: DAI, USDC, USDT         │
    │                     Rate = 10^(18-decimals)                    │
    │                                                                 │
    │  get_dy_underlying(): Uses precision multipliers                │
    │                      For yield-bearing tokens                  │
    │                      Precision = 10^(18-decimals)              │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Use Cases for Underlying Calculations:
    
    1. Compound Integration (cTokens):
    ┌─────────────────────────────────────────────────────────────────┐
    │  User has: 100 USDC                                            │
    │  Pool has: cUSDC (earning interest)                            │
    │  Question: How much cDAI will I get for 100 underlying USDC?   │
    │  Answer: get_dy_underlying(USDC_index, DAI_index, 100e6)       │
    └─────────────────────────────────────────────────────────────────┘
    
    2. Aave Integration (aTokens):
    ┌─────────────────────────────────────────────────────────────────┐
    │  User wants: 50 underlying DAI                                 │
    │  Pool has: aDAI, aUSDC, aUSDT                                  │
    │  Question: How much underlying USDC needed?                    │
    │  Answer: Use underlying amounts, not aToken amounts            │
    └─────────────────────────────────────────────────────────────────┘
    
    Mathematical Process (Same as get_dy but different normalization):
    ┌─────────────────────────────────────────────────────────────────┐
    │                 Underlying Asset Calculation                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Normalize using PRECISION_MUL instead of RATES             │
    │  2. Calculate new balance after input                          │
    │  3. Solve invariant for new output balance                     │
    │  4. Convert back using PRECISION_MUL                           │
    │  5. Apply trading fees                                         │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Precision Handling Comparison:
    ┌─────────────────────────────────────────────────────────────────┐
    │  Token     │  Decimals  │  RATES        │  PRECISION_MUL        │
    │──────────────────────────────────────────────────────────────────│
    │  DAI       │     18     │    1e18       │       1e0             │
    │  USDC      │      6     │    1e30       │       1e12            │
    │  USDT      │      6     │    1e30       │       1e12            │
    └─────────────────────────────────────────────────────────────────┘
    
    Example Calculation:
    Swap 100 underlying USDC → underlying DAI
    
    Step 1: Normalize input
    - dx = 100 * 1e6 (USDC native units)
    - dx_normalized = 100e6 * 1e12 = 100e18
    
    Step 2-5: Same as get_dy()
    - Solve invariant, calculate output, apply fees
    - Result: ~99.84e18 (normalized)
    
    Step 6: Convert to underlying DAI
    - Output = 99.84e18 / 1e0 = 99.84e18 DAI
    
    Real-World Integration Example:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Compound cToken Pool                        │
    │                                                                 │
    │  Pool contains: [cDAI, cUSDC, cUSDT]                           │
    │  Exchange rates: [0.021, 0.022, 0.020] (underlying per cToken) │
    │                                                                 │
    │  User query: "I have 1000 USDC, how much DAI will I get?"     │
    │  Answer: get_dy_underlying(1, 0, 1000e6) = 998.5e18           │
    │                                                                 │
    │  This accounts for:                                            │
    │  - USDC → cUSDC conversion at current rate                     │
    │  - cUSDC → cDAI swap through StableSwap                       │
    │  - cDAI → DAI conversion at current rate                      │
    │  - Trading fees and slippage                                  │
    └─────────────────────────────────────────────────────────────────┘
    
    When to Use:
    - ✅ Yield-bearing asset pools (Compound, Aave)
    - ✅ Frontend interfaces showing underlying amounts
    - ✅ Cross-protocol integrations
    - ✅ User-friendly calculations
    
    When NOT to Use:
    - ❌ Standard token pools (use get_dy instead)
    - ❌ Internal contract calculations
    - ❌ When you need exact token amounts
    
    @param i Index of input token
    @param j Index of output token  
    @param dx Amount of input token in underlying units
    @return Amount of output token in underlying units (after fees)
    """
    # Get current normalized balances and precision multipliers
    xp: uint256[N_COINS] = self._xp()
    precisions: uint256[N_COINS] = PRECISION_MUL

    # Convert input to normalized precision using precision multipliers
    x: uint256 = xp[i] + dx * precisions[i]
    
    # Solve StableSwap invariant for new output balance
    y: uint256 = self.get_y(i, j, x, xp)
    
    # Calculate raw output and convert back to underlying units
    # Subtract 1 for precision safety, then divide by precision multiplier
    dy: uint256 = (xp[j] - y - 1) / precisions[j]
    
    # Apply trading fee
    _fee: uint256 = self.fee * dy / FEE_DENOMINATOR
    return dy - _fee



@external
@nonreentrant('lock')
def exchange(i: int128, j: int128, dx: uint256, min_dy: uint256):
    """
    @notice Execute a token swap between two assets in the pool
    @dev This is the core swapping function that maintains the StableSwap invariant
         while transferring tokens and applying fees. Includes protection against
         fee-on-transfer tokens and reentrancy attacks.
    
    Complete Swap Process Flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Token Exchange Process                    │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Security Checks: Pool active, valid indices                │
    │  2. Transfer Input: Handle fee-on-transfer tokens (USDT)       │
    │  3. Calculate Output: Solve StableSwap invariant               │
    │  4. Apply Fees: Trading fee + admin fee split                  │
    │  5. Update Balances: Maintain pool accounting                  │
    │  6. Transfer Output: Send tokens to user                       │
    │  7. Emit Event: Log transaction for indexing                   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Mathematical Process:
    Given input amount dx of token i, calculate output amount dy of token j
    
    1. New input balance: x_new = x_old + dx
    2. Solve invariant: D = constant = 4Ax∑xᵢ + ∏xᵢ
    3. Find y_new such that D remains constant
    4. Raw output: dy_raw = y_old - y_new
    5. Apply fee: dy_final = dy_raw * (1 - fee_rate)
    
    Fee Structure Visualization:
    ┌─────────────────────────────────────────────────────────────────┐
    │                         Fee Distribution                        │
    │                                                                 │
    │  Total Trade Value: 100 tokens                                 │
    │       │                                                         │
    │       ├─► 99.96 tokens ─► User (99.96%)                        │
    │       │                                                         │
    │       └─► 0.04 tokens ─► Trading Fee (0.04%)                   │
    │               │                                                 │
    │               ├─► 0.02 tokens ─► LP Providers (50%)            │
    │               └─► 0.02 tokens ─► Admin/DAO (50%)               │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Fee-on-Transfer Token Handling (USDT):
    ┌─────────────────────────────────────────────────────────────────┐
    │                    USDT Special Handling                       │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Standard Token:                                               │
    │  - transferFrom(user, pool, 100) → pool receives exactly 100   │
    │                                                                 │
    │  USDT (Fee-on-Transfer):                                       │
    │  - transferFrom(user, pool, 100) → pool receives ~99.9        │
    │  - Internal fee: ~0.1 tokens                                   │
    │                                                                 │
    │  Protection Method:                                            │
    │  1. Record balance before transfer                             │
    │  2. Execute transferFrom()                                     │
    │  3. Check actual balance increase                              │
    │  4. Use actual received amount for calculations                │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Example Swap: 100 USDC → DAI
    
    Initial State:
    - Pool: [1000 DAI, 1000 USDC, 1000 USDT]
    - User: 100 USDC to swap
    - Fee: 0.04% = 4e6/1e10
    
    Step 1: Transfer 100 USDC to pool
    - New USDC balance: 1100 USDC
    
    Step 2: Calculate StableSwap
    - Normalize: [1000e18, 1100e18, 1000e18]
    - Solve for new DAI balance: ~900.12e18
    - Raw output: 1000e18 - 900.12e18 = 99.88e18
    
    Step 3: Apply fees
    - Trading fee: 99.88 * 0.04% = 0.04 DAI
    - User receives: 99.88 - 0.04 = 99.84 DAI
    - Admin fee: 0.04 * 50% = 0.02 DAI
    - LP providers: 0.02 DAI (added to pool)
    
    Step 4: Update balances
    - Pool DAI: 1000 - 99.84 - 0.02 = 900.14 DAI
    - Pool USDC: 1100 USDC
    - Pool USDT: 1000 USDT (unchanged)
    
    Security Features:
    ┌─────────────────────────────────────────────────────────────────┐
    │                       Security Measures                        │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  ✅ Reentrancy Protection: @nonreentrant('lock')               │
    │  ✅ Pool Kill Switch: assert not self.is_killed               │
    │  ✅ Slippage Protection: min_dy parameter                      │
    │  ✅ Safe Transfers: Custom ERC20 handling                     │
    │  ✅ Fee-on-Transfer: Special USDT handling                    │
    │  ✅ Precision Safety: -1 wei for rounding errors              │
    │  ✅ Balance Validation: Exact accounting                      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Error Conditions:
    - Pool is killed/paused
    - Insufficient output (slippage too high)  
    - Token transfer failures
    - Invalid token indices
    - Zero amounts
    
    Gas Optimization Notes:
    - Uses memory arrays for calculations
    - Minimizes storage reads/writes
    - Efficient precision handling
    - Single token transfer per direction
    
    @param i Index of input token being sold (0=DAI, 1=USDC, 2=USDT)
    @param j Index of output token being bought (0=DAI, 1=USDC, 2=USDT)
    @param dx Amount of input token to sell (in token's native units)
    @param min_dy Minimum output amount expected (slippage protection)
    """
    # Ensure pool is active and not in emergency shutdown
    assert not self.is_killed  # dev: is killed
    
    # Get precision rate multipliers for token normalization
    rates: uint256[N_COINS] = RATES 

    # Capture current pool state before any changes
    old_balances: uint256[N_COINS] = self.balances
    xp: uint256[N_COINS] = self._xp_mem(old_balances)

    # Handle fee-on-transfer tokens (USDT has index FEE_INDEX)
    # Some tokens charge fees on transfers, reducing the actual received amount
    dx_w_fee: uint256 = dx
    input_coin: address = self.coins[i]

    # Record balance before transfer for fee-on-transfer token detection
    if i == FEE_INDEX:
        dx_w_fee = ERC20(input_coin).balanceOf(self)

    # Execute safe token transfer from user to pool
    # Custom implementation handles tokens that don't return transfer result
    _response: Bytes[32] = raw_call(
        input_coin,
        concat(
            method_id("transferFrom(address,address,uint256)"),
            convert(msg.sender, bytes32),
            convert(self,  bytes32),
            convert(dx, bytes32),
        ),
        max_outsize=32,
    )  # dev: failed transfer
    
    # Check transfer success for tokens that return boolean result
    if len(_response) > 0:
        assert convert(_response, bool)  # dev: failed transfer

    # Calculate actual received amount for fee-on-transfer tokens
    if i == FEE_INDEX:
        dx_w_fee = ERC20(input_coin).balanceOf(self) - dx_w_fee

    # Convert input to normalized precision and calculate new input balance
    x: uint256 = xp[i] + dx_w_fee * rates[i] / PRECISION
    
    # Solve StableSwap invariant to find new output token balance
    y: uint256 = self.get_y(i, j, x, xp)

    # Calculate raw output amount (subtract 1 wei for precision safety)
    dy: uint256 = xp[j] - y - 1

    # Apply trading fee to the output amount
    dy_fee: uint256 = dy * self.fee / FEE_DENOMINATOR

    # Convert output to token's native precision and subtract fees
    dy = (dy - dy_fee) * PRECISION / rates[j]
    
    # Slippage protection: ensure user receives minimum expected amount
    assert dy >= min_dy, "Exchange resulted in fewer coins than expected"

    # Calculate admin fee portion (typically 50% of trading fees)
    dy_admin_fee: uint256 = dy_fee * self.admin_fee / FEE_DENOMINATOR
    dy_admin_fee = dy_admin_fee * PRECISION / rates[j]

    # Update pool balances:
    # - Increase input token balance by actual received amount
    # - Decrease output token balance by user amount + admin fee
    self.balances[i] = old_balances[i] + dx_w_fee
    self.balances[j] = old_balances[j] - dy - dy_admin_fee

    # Transfer output tokens to user using safe transfer method
    _response = raw_call(
        self.coins[j],
        concat(
            method_id("transfer(address,uint256)"),
            convert(msg.sender, bytes32),
            convert(dy, bytes32),
        ),
        max_outsize=32,
    )  # dev: failed transfer
    
    # Verify transfer success
    if len(_response) > 0:
        assert convert(_response, bool)  # dev: failed transfer

    # Emit event for off-chain tracking and analytics
    log TokenExchange(msg.sender, i, dx, j, dy)


@external
@nonreentrant('lock')
def remove_liquidity(_amount: uint256, min_amounts: uint256[N_COINS]):
    """
    @notice Remove liquidity from pool in balanced proportions
    @dev Burns LP tokens and returns all underlying assets proportionally
         This is the "balanced withdrawal" method that maintains pool ratios
    
    Balanced Withdrawal Process:
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Balanced Liquidity Removal                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Calculate proportional share of each token                  │
    │  2. Validate minimum amounts (slippage protection)             │
    │  3. Update pool balances                                        │
    │  4. Transfer tokens to user                                     │
    │  5. Burn LP tokens from user                                    │
    │  6. Emit removal event                                          │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Mathematical Foundation:
    Each user owns a proportional share of the entire pool based on LP tokens
    
    withdrawal_amount[i] = pool_balance[i] * LP_tokens_burned / total_LP_supply
    
    Example Calculation:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Withdrawal Example                         │
    │                                                                 │
    │  Pool State:                                                    │
    │  - DAI: 1,000,000 tokens                                       │
    │  - USDC: 1,200,000 tokens                                      │
    │  - USDT: 800,000 tokens                                        │
    │  - Total LP tokens: 2,950,000                                  │
    │                                                                 │
    │  User wants to withdraw: 29,500 LP tokens (1% of pool)        │
    │                                                                 │
    │  Calculations:                                                  │
    │  - DAI out: 1,000,000 * 29,500 / 2,950,000 = 10,000 DAI      │
    │  - USDC out: 1,200,000 * 29,500 / 2,950,000 = 12,000 USDC    │
    │  - USDT out: 800,000 * 29,500 / 2,950,000 = 8,000 USDT       │
    │                                                                 │
    │  Result: User receives exactly 1% of each token in pool       │
    └─────────────────────────────────────────────────────────────────┘
    
    Visual Representation:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Pool Composition                            │
    │                                                                 │
    │   Before Withdrawal:          After Withdrawal:               │
    │   ┌─────────────────┐         ┌─────────────────┐             │
    │   │   Pool Assets   │         │   Pool Assets   │             │
    │   │  DAI:   1000k   │   ──►   │  DAI:    990k   │             │
    │   │  USDC:  1200k   │         │  USDC:  1188k   │             │
    │   │  USDT:   800k   │         │  USDT:   792k   │             │
    │   │  LP:    2950k   │         │  LP:    2920.5k │             │
    │   └─────────────────┘         └─────────────────┘             │
    │                                                                 │
    │   User receives exactly proportional amounts                   │
    │   maintaining the same pool composition ratios                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Key Properties:
    1. **Proportional**: Always maintains pool token ratios
    2. **No Fees**: Balanced withdrawals don't pay trading fees
    3. **No Slippage**: Price impact is zero for balanced removal
    4. **Fair Value**: User receives exact pro-rata share
    5. **Gas Efficient**: Simple calculation, no complex math
    
    Comparison with Other Withdrawal Methods:
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Withdrawal Method Comparison                  │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Balanced (this function):                                     │
    │  ✅ No fees, no slippage                                       │
    │  ✅ Guaranteed fair value                                      │
    │  ❌ Must accept all tokens                                     │
    │  ❌ Cannot choose specific assets                              │
    │                                                                 │
    │  Imbalanced (remove_liquidity_imbalance):                     │
    │  ✅ Choose specific token amounts                              │
    │  ❌ Pays fees for imbalance                                   │
    │  ❌ More complex calculations                                  │
    │                                                                 │
    │  Single Token (remove_liquidity_one_coin):                    │
    │  ✅ Receive only one desired token                            │
    │  ❌ Highest fees and slippage                                 │
    │  ❌ Most expensive option                                     │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Use Cases:
    1. **Portfolio Rebalancing**: Exit entire position proportionally
    2. **Liquidity Management**: Reduce exposure while maintaining ratios
    3. **Risk Management**: Emergency exit without trading costs
    4. **Arbitrage Prevention**: No MEV opportunities in balanced withdrawals
    5. **Gas Optimization**: Most efficient withdrawal method
    
    Security Features:
    - ✅ Reentrancy protection
    - ✅ Minimum amount validation (slippage protection)
    - ✅ Safe token transfers
    - ✅ Exact balance accounting
    - ✅ LP token burn verification
    
    Edge Cases Handled:
    - Zero total supply (should never happen after initialization)
    - Rounding errors (favors pool slightly)
    - Token transfer failures
    - Insufficient LP token balance
    
    @param _amount Number of LP tokens to burn for withdrawal
    @param min_amounts Minimum amounts of each token to receive (slippage protection)
    """
    # Get current total LP token supply
    total_supply: uint256 = self.token.totalSupply()
    
    # Initialize arrays for withdrawal amounts and fees (fees unused but kept for event)
    amounts: uint256[N_COINS] = empty(uint256[N_COINS])
    fees: uint256[N_COINS] = empty(uint256[N_COINS])  # Unused but preserved for historical events

    # Calculate proportional withdrawal for each token
    for i in range(N_COINS):
        # Calculate user's proportional share: balance * LP_tokens / total_supply
        value: uint256 = self.balances[i] * _amount / total_supply
        
        # Slippage protection: ensure minimum amounts are met
        assert value >= min_amounts[i], "Withdrawal resulted in fewer coins than expected"
        
        # Update pool balance by removing withdrawn amount
        self.balances[i] -= value
        amounts[i] = value

        # Execute safe token transfer to user
        _response: Bytes[32] = raw_call(
            self.coins[i],
            concat(
                method_id("transfer(address,uint256)"),
                convert(msg.sender, bytes32),
                convert(value, bytes32),
            ),
            max_outsize=32,
        )  # dev: failed transfer
        
        # Verify transfer success for tokens that return boolean
        if len(_response) > 0:
            assert convert(_response, bool)  # dev: failed transfer

    # Burn LP tokens from user (will fail if insufficient balance)
    self.token.burnFrom(msg.sender, _amount)  # dev: insufficient funds

    # Emit event for off-chain tracking and analytics
    log RemoveLiquidity(msg.sender, amounts, fees, total_supply - _amount)


@external
@nonreentrant('lock')
def remove_liquidity_imbalance(amounts: uint256[N_COINS], max_burn_amount: uint256):
    """
    @notice Remove specific amounts of tokens from pool (imbalanced withdrawal)
    @dev Allows users to withdraw exact amounts of specific tokens, paying fees
         for any imbalance created. More expensive than balanced withdrawal but
         offers precise control over received token amounts.
    
    Imbalanced Withdrawal Process:
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Imbalanced Liquidity Removal                  │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Calculate current pool invariant D₀                        │
    │  2. Simulate withdrawal: calculate new balances                │
    │  3. Calculate post-withdrawal invariant D₁                     │
    │  4. Calculate imbalance fees for deviation from ideal          │
    │  5. Apply fees: get final invariant D₂                         │
    │  6. Calculate LP tokens to burn: (D₀ - D₂) / D₀ * supply       │
    │  7. Execute transfers and burn LP tokens                       │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Mathematical Foundation:
    When users withdraw specific amounts, they create an imbalance that 
    deviates from the ideal proportional withdrawal. This imbalance is
    penalized through fees to prevent arbitrage.
    
    Fee Calculation Process:
    ┌─────────────────────────────────────────────────────────────────┐
    │                       Fee Structure                             │
    │                                                                 │
    │  1. Calculate ideal balanced withdrawal for same D change       │
    │  2. Compare actual vs ideal balances                           │
    │  3. Apply fee based on deviation magnitude                     │
    │                                                                 │
    │  fee_rate = base_fee * N_COINS / (4 * (N_COINS - 1))          │
    │  fee_rate = 0.04% * 3 / (4 * 2) = 0.015% (reduced fee)        │
    │                                                                 │
    │  For each token:                                               │
    │  ideal_balance[i] = new_D * old_balance[i] / old_D             │
    │  deviation[i] = |actual_balance[i] - ideal_balance[i]|         │
    │  fee[i] = fee_rate * deviation[i]                              │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Example Imbalanced Withdrawal:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Withdrawal Example                        │
    │                                                                 │
    │  Initial Pool State:                                           │
    │  - DAI: 1,000,000                                             │
    │  - USDC: 1,000,000                                            │
    │  - USDT: 1,000,000                                            │
    │  - D₀ = 3,000,000 (approximately)                             │
    │  - Total LP: 2,950,000                                        │
    │                                                                 │
    │  User wants to withdraw: [200,000 DAI, 0 USDC, 0 USDT]       │
    │                                                                 │
    │  Step 1: Calculate D₁ (after withdrawal, before fees)         │
    │  New balances: [800,000, 1,000,000, 1,000,000]               │
    │  D₁ = 2,810,000 (approximately)                               │
    │                                                                 │
    │  Step 2: Calculate ideal balanced withdrawal                   │
    │  For D change of (3,000,000 - 2,810,000) = 190,000           │
    │  Ideal withdrawal: [63,333 DAI, 63,333 USDC, 63,333 USDT]    │
    │  Ideal balances: [936,667, 936,667, 936,667]                 │
    │                                                                 │
    │  Step 3: Calculate imbalance fees                             │
    │  Deviations: [136,667, 63,333, 63,333]                       │
    │  Fees: [205 DAI, 95 USDC, 95 USDT]                          │
    │                                                                 │
    │  Step 4: Final state after fees                              │
    │  D₂ = 2,800,000 (approximately)                               │
    │  LP burned = (3,000,000 - 2,800,000) * 2,950,000 / 3,000,000 │
    │            = 196,667 LP tokens                                │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Visual Representation of Imbalance:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Pool Balance Changes                        │
    │                                                                 │
    │   Ideal Withdrawal:            Actual Withdrawal:              │
    │   ┌─────────────────┐         ┌─────────────────┐             │
    │   │ Before  │ After │         │ Before  │ After │             │
    │   │ DAI: 1M │ 937k  │         │ DAI: 1M │ 800k  │ ← More     │
    │   │ USDC:1M │ 937k  │         │ USDC:1M │ 1M    │ ← Same     │
    │   │ USDT:1M │ 937k  │         │ USDT:1M │ 1M    │ ← Same     │
    │   └─────────────────┘         └─────────────────┘             │
    │                                                                 │
    │   Balanced (no fees)           Imbalanced (fees apply)        │
    │   Pool remains symmetric       Pool becomes asymmetric        │
    └─────────────────────────────────────────────────────────────────┘
    
    Fee Comparison:
    ┌─────────────────────────────────────────────────────────────────┐
    │              Withdrawal Method Fee Comparison                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Balanced Withdrawal:                                          │
    │  ✅ Fee: 0% (no trading fees)                                  │
    │  ✅ Cost: Only gas                                            │
    │                                                                 │
    │  Imbalanced Withdrawal:                                        │
    │  🟡 Fee: 0.015% × imbalance_amount                             │
    │  🟡 Cost: Gas + imbalance fees                                │
    │                                                                 │
    │  Single Token Withdrawal:                                      │
    │  🔴 Fee: 0.04% × full_amount                                   │
    │  🔴 Cost: Gas + full trading fees                             │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Use Cases:
    1. **Targeted Rebalancing**: Remove specific tokens for portfolio adjustment
    2. **Debt Repayment**: Get exact amounts needed for loan payments
    3. **Arbitrage**: Extract specific tokens for cross-protocol opportunities
    4. **Risk Management**: Reduce exposure to particular assets
    5. **Strategic Exit**: Optimize tax implications or timing
    
    Security Considerations:
    ┌─────────────────────────────────────────────────────────────────┐
    │                       Security Features                        │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  ✅ Pool Kill Switch: Prevents withdrawals during emergencies  │
    │  ✅ Non-Zero Supply: Ensures pool has liquidity               │
    │  ✅ Slippage Protection: max_burn_amount parameter             │
    │  ✅ Rounding Protection: +1 token to prevent exploits         │
    │  ✅ Safe Transfers: Custom ERC20 handling                     │
    │  ✅ Exact Accounting: Precise balance updates                 │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Gas Optimization Notes:
    - Batches all token transfers
    - Minimizes storage operations
    - Single LP token burn operation
    - Efficient memory usage for calculations
    
    @param amounts Array of exact token amounts to withdraw [DAI, USDC, USDT]
    @param max_burn_amount Maximum LP tokens willing to burn (slippage protection)
    """
    # Ensure pool is active and not in emergency shutdown
    assert not self.is_killed  # dev: is killed

    # Validate pool has liquidity
    token_supply: uint256 = self.token.totalSupply()
    assert token_supply != 0  # dev: zero total supply
    
    # Calculate reduced fee rate for imbalanced liquidity removal
    # This is 1/4 of the swap fee, making imbalanced withdrawals less penalized
    _fee: uint256 = self.fee * N_COINS / (4 * (N_COINS - 1))
    _admin_fee: uint256 = self.admin_fee
    amp: uint256 = self._A()

    # Capture current pool state
    old_balances: uint256[N_COINS] = self.balances
    new_balances: uint256[N_COINS] = old_balances
    
    # Calculate current pool invariant
    D0: uint256 = self.get_D_mem(old_balances, amp)
    
    # Simulate withdrawal: calculate new balances after removing requested amounts
    for i in range(N_COINS):
        new_balances[i] -= amounts[i]
    
    # Calculate invariant after withdrawal (before fees)
    D1: uint256 = self.get_D_mem(new_balances, amp)
    
    # Calculate imbalance fees
    fees: uint256[N_COINS] = empty(uint256[N_COINS])
    for i in range(N_COINS):
        # Calculate ideal balance: what this token's balance should be
        # if the withdrawal was perfectly balanced
        ideal_balance: uint256 = D1 * old_balances[i] / D0
        
        # Calculate deviation from ideal balance
        difference: uint256 = 0
        if ideal_balance > new_balances[i]:
            difference = ideal_balance - new_balances[i]  # Under-withdrawn
        else:
            difference = new_balances[i] - ideal_balance  # Over-withdrawn
        
        # Calculate imbalance fee for this token
        fees[i] = _fee * difference / FEE_DENOMINATOR
        
        # Update pool balance: subtract withdrawn amount and admin fee portion
        self.balances[i] = new_balances[i] - (fees[i] * _admin_fee / FEE_DENOMINATOR)
        
        # Subtract fee from calculation balance for final invariant
        new_balances[i] -= fees[i]
    
    # Calculate final invariant after fee deduction
    D2: uint256 = self.get_D_mem(new_balances, amp)

    # Calculate LP tokens to burn: proportional to invariant decrease
    # LP_burned = (D₀ - D₂) / D₀ * total_supply
    token_amount: uint256 = (D0 - D2) * token_supply / D0
    assert token_amount != 0  # dev: zero tokens burned
    
    # Add 1 wei to protect against rounding attacks (favors pool)
    token_amount += 1
    
    # Slippage protection: ensure user doesn't burn more LP tokens than expected
    assert token_amount <= max_burn_amount, "Slippage screwed you"

    # Burn LP tokens from user
    self.token.burnFrom(msg.sender, token_amount)  # dev: insufficient funds
    
    # Transfer requested token amounts to user
    for i in range(N_COINS):
        if amounts[i] != 0:
            # Execute safe token transfer
            _response: Bytes[32] = raw_call(
                self.coins[i],
                concat(
                    method_id("transfer(address,uint256)"),
                    convert(msg.sender, bytes32),
                    convert(amounts[i], bytes32),
                ),
                max_outsize=32,
            )  # dev: failed transfer
            
            # Verify transfer success
            if len(_response) > 0:
                assert convert(_response, bool)  # dev: failed transfer

    # Emit event for off-chain tracking
    log RemoveLiquidityImbalance(msg.sender, amounts, fees, D1, token_supply - token_amount)


@view
@internal
def get_y_D(A_: uint256, i: int128, xp: uint256[N_COINS], D: uint256) -> uint256:
    """
    @notice Calculate balance of token i when pool invariant is reduced to a specific value D
    @dev This function is the mathematical core of single-token withdrawals. It solves
         the StableSwap invariant equation for a specific token balance when the total
         invariant D is known (typically reduced due to LP token burning).
    
    Mathematical Foundation:
    We need to solve the StableSwap invariant for x[i] when D and all other balances are known:
    
    An³∑xⱼ + D = ADn³ + D⁴/(4³∏xⱼ)
    
    Rearranging for token i when all other tokens j≠i are fixed:
    xᵢ² + xᵢ(b - D) - c = 0
    
    Where:
    - b = S + D/A  (S = sum of all other token balances except i)
    - c = D³/(An³∏xⱼ) for j≠i
    - A = amplification coefficient
    - D = target invariant (reduced from original)
    
    Newton's Method Solution:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Quadratic Equation Solver                   │
    │                                                                 │
    │  f(x) = x² + x(b-D) - c = 0                                    │
    │  f'(x) = 2x + (b-D)                                           │
    │                                                                 │
    │  Newton's iteration:                                           │
    │  x_new = x_old - f(x_old)/f'(x_old)                          │
    │        = x_old - (x_old² + x_old(b-D) - c)/(2x_old + b-D)    │
    │        = (x_old² + c)/(2x_old + b - D)                       │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Convergence Visualization:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Newton's Method Convergence                     │
    │                                                                 │
    │     f(x)                                                       │
    │      │                                                         │
    │      │     ╭─╮  Quadratic curve                               │
    │      │   ╭─╯   ╲                                               │
    │      │ ╭─╯       ╲                                             │
    │  ────┼─╯───────────╲──────────────► x (token balance)         │
    │      │               ╲                                         │
    │      │                 ╲                                       │
    │      │                   ╲                                     │
    │                            ╲                                   │
    │    x₀     x₁    x₂    x₃    ╲ ← Root (solution)               │
    │    ↑      ↑     ↑     ↑                                       │
    │    Start  Iter1 Iter2 Final                                   │
    │                                                                 │
    │  Each iteration gets closer to where f(x) = 0                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Example Single Token Withdrawal Calculation:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Withdrawal Example                          │
    │                                                                 │
    │  Initial Pool State:                                           │
    │  - DAI: 1,000,000 (xp[0] = 1000e18)                          │
    │  - USDC: 1,000,000 (xp[1] = 1000e18)                         │
    │  - USDT: 1,000,000 (xp[2] = 1000e18)                         │
    │  - D₀ = 3,000,000 (approximately)                             │
    │  - Total LP: 2,950,000                                        │
    │                                                                 │
    │  User wants to burn: 295,000 LP tokens (10% of supply)       │
    │  Target token: DAI (i = 0)                                    │
    │                                                                 │
    │  Step 1: Calculate target invariant                           │
    │  D₁ = D₀ - (295,000 * D₀ / 2,950,000)                        │
    │     = 3,000,000 - 300,000 = 2,700,000                        │
    │                                                                 │
    │  Step 2: Set up equation for DAI balance                      │
    │  Fixed balances: USDC = 1000e18, USDT = 1000e18              │
    │  S = 1000e18 + 1000e18 = 2000e18                             │
    │  b = S + D₁/A = 2000e18 + 2,700,000e18/100 = 2027e18        │
    │  c = D₁³/(A×n³×USDC×USDT)                                     │
    │    = (2.7e6)³/(100×27×1000e18×1000e18) ≈ 726e18              │
    │                                                                 │
    │  Step 3: Solve x² + x(2027e18 - 2.7e6e18) - 726e18 = 0      │
    │  Using Newton's method: converges to x ≈ 700e18              │
    │                                                                 │
    │  Step 4: Calculate withdrawal amount                          │
    │  DAI withdrawn = 1000e18 - 700e18 = 300e18 = 300,000 DAI    │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Use Cases in Single Token Withdrawals:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Function Applications                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Calculate withdrawal amount (without fees)                 │
    │     - Input: LP tokens to burn, target token                  │
    │     - Output: Raw token amount before fees                    │
    │                                                                 │
    │  2. Calculate withdrawal amount (with fees)                    │
    │     - Apply imbalance fees to the raw amount                  │
    │     - Account for reduced pool balance after fees             │
    │                                                                 │
    │  3. Validate withdrawal feasibility                           │
    │     - Ensure sufficient token balance exists                  │
    │     - Check that solution converges properly                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Comparison with Other Balance Calculations:
    ┌─────────────────────────────────────────────────────────────────┐
    │              get_y() vs get_y_D() Comparison                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  get_y() - Swap calculations:                                  │
    │  ✓ Input: New balance of input token                          │
    │  ✓ Output: New balance of output token                        │
    │  ✓ Maintains constant D (invariant preservation)              │
    │  ✓ Used for: Token swaps, price calculations                  │
    │                                                                 │
    │  get_y_D() - Withdrawal calculations:                         │
    │  ✓ Input: Target invariant D (reduced from original)          │
    │  ✓ Output: New balance of withdrawal token                    │
    │  ✓ Allows D to change (liquidity removal)                     │
    │  ✓ Used for: Single token withdrawals, LP burning             │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Mathematical Properties:
    1. **Uniqueness**: For valid inputs, there's exactly one positive solution
    2. **Convergence**: Newton's method typically converges in 3-8 iterations
    3. **Stability**: Small changes in D produce proportional changes in result
    4. **Bounded**: Solution is always less than the sum of other token balances
    
    @param A_ Amplification coefficient (current value, not interpolated)
    @param i Index of token to solve for (0=DAI, 1=USDC, 2=USDT)
    @param xp Array of current normalized token balances
    @param D Target invariant value (typically reduced from current)
    @return New balance of token i that satisfies the target invariant D
    """
    # Validate token index to prevent invalid calculations
    assert i >= 0  # dev: i below zero
    assert i < N_COINS  # dev: i above N_COINS

    # Initialize variables for building the quadratic equation
    # c will become: c = D³/(An³∏xⱼ) for j≠i
    c: uint256 = D
    S_: uint256 = 0  # Sum of balances for all tokens except i
    Ann: uint256 = A_ * N_COINS  # A * n for efficiency

    # Build the equation coefficients by iterating through all tokens except i
    _x: uint256 = 0
    for _i in range(N_COINS):
        if _i != i:
            # Use current balance for all tokens except the target token i
            _x = xp[_i]
        else:
            # Skip the target token i - we're solving for its balance
            continue
        
        # Add to sum of other token balances
        S_ += _x
        
        # Build c coefficient: multiply by D and divide by (_x * N_COINS)
        # This constructs: c = c * D / (_x * N_COINS) = D * D * D / (∏_x * N_COINS³)
        c = c * D / (_x * N_COINS)
    
    # Complete the c calculation: c = D³/(Ann * N_COINS) * 1/∏xⱼ
    c = c * D / (Ann * N_COINS)
    
    # Calculate b coefficient: b = S + D/Ann
    # This represents: S = sum of other balances, D/Ann = D/(A*n)
    b: uint256 = S_ + D / Ann
    
    # Newton's method iteration to solve: y² + y(b-D) - c = 0
    y_prev: uint256 = 0
    y: uint256 = D  # Initial guess: start with D as the balance estimate
    
    # Iterate using Newton's method with maximum 255 iterations for safety
    for _i in range(255):
        y_prev = y
        
        # Newton's method update: y = (y² + c) / (2y + b - D)
        # This comes from: y_new = y_old - f(y_old)/f'(y_old)
        # Where f(y) = y² + y(b-D) - c and f'(y) = 2y + (b-D)
        y = (y*y + c) / (2 * y + b - D)
        
        # Check convergence (precision of 1 wei)
        if y > y_prev:
            if y - y_prev <= 1:
                break  # Converged from below
        else:
            if y_prev - y <= 1:
                break  # Converged from above
    
    return y


@view
@internal
def _calc_withdraw_one_coin(_token_amount: uint256, i: int128) -> (uint256, uint256):
    """
    @notice Calculate withdrawal amount and fees for single-token withdrawal
    @dev This is the core calculation engine for single-token withdrawals. It computes
         both the gross withdrawal amount (before fees) and the net amount (after fees)
         that the user will receive when burning LP tokens for a single asset.
    
    Single Token Withdrawal Process:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Single Token Withdrawal Flow                    │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. Calculate target invariant D₁ after LP token burn          │
    │  2. Solve for new token balance using get_y_D()                │
    │  3. Calculate gross withdrawal (before fees)                   │
    │  4. Calculate imbalance fees for the withdrawal                │
    │  5. Apply fees to get net withdrawal amount                    │
    │  6. Return both net amount and fee amount                      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Mathematical Foundation:
    
    When a user burns LP tokens and wants only one type of asset, they create
    an imbalance in the pool. This deviation from the ideal balanced withdrawal
    is penalized through fees to prevent arbitrage exploitation.
    
    Step 1: Calculate New Invariant
    D₁ = D₀ - (LP_burned / total_LP_supply) × D₀
    D₁ = D₀ × (1 - LP_burned / total_LP_supply)
    
    Step 2: Ideal vs Actual Withdrawal
    ┌─────────────────────────────────────────────────────────────────┐
    │                 Ideal vs Actual Comparison                     │
    │                                                                 │
    │  Ideal Balanced Withdrawal:    Actual Single Token Withdrawal: │
    │  ┌─────────────────┐          ┌─────────────────┐             │
    │  │ Before │ After  │          │ Before │ After  │             │
    │  │ DAI:1000│ 900   │          │ DAI:1000│ 700   │ ← More     │
    │  │ USDC:1000│ 900  │          │ USDC:1000│ 1000 │ ← Same     │
    │  │ USDT:1000│ 900  │          │ USDT:1000│ 1000 │ ← Same     │
    │  └─────────────────┘          └─────────────────┘             │
    │                                                                 │
    │  No imbalance (no fees)       Large imbalance (fees apply)    │
    └─────────────────────────────────────────────────────────────────┘
    
    Step 3: Fee Calculation Process
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Fee Structure                              │
    │                                                                 │
    │  Base Fee Rate:                                                │
    │  fee_rate = pool_fee × N_COINS / (4 × (N_COINS-1))            │
    │  fee_rate = 0.04% × 3 / (4 × 2) = 0.015%                      │
    │                                                                 │
    │  For each token j:                                             │
    │  1. Calculate ideal balance after D change                     │
    │     ideal[j] = current[j] × D₁ / D₀                           │
    │                                                                 │
    │  2. Calculate actual balance after withdrawal                  │
    │     actual[j] = current[j] - withdrawal_amount (if j=i)        │
    │     actual[j] = current[j] (if j≠i)                           │
    │                                                                 │
    │  3. Calculate deviation and apply fee                         │
    │     deviation[j] = |actual[j] - ideal[j]|                     │
    │     fee[j] = fee_rate × deviation[j]                          │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Detailed Example Calculation:
    ┌─────────────────────────────────────────────────────────────────┐
    │                   Withdrawal Calculation                       │
    │                                                                 │
    │  Initial State:                                                │
    │  - Pool: [1000k DAI, 1000k USDC, 1000k USDT]                 │
    │  - D₀ = 3,000,000 (approximately)                             │
    │  - Total LP = 2,950,000                                       │
    │  - Pool fee = 0.04%                                           │
    │                                                                 │
    │  User Action:                                                  │
    │  - Burns: 295,000 LP tokens (10% of supply)                   │
    │  - Wants: Only DAI (i = 0)                                    │
    │                                                                 │
    │  Step 1: Calculate new invariant                              │
    │  D₁ = 3,000,000 × (1 - 295,000/2,950,000) = 2,700,000        │
    │                                                                 │
    │  Step 2: Calculate gross withdrawal (before fees)             │
    │  Using get_y_D(): new_DAI_balance ≈ 700,000                   │
    │  Gross withdrawal = 1,000,000 - 700,000 = 300,000 DAI        │
    │                                                                 │
    │  Step 3: Calculate ideal balances                             │
    │  ideal_DAI = 1,000,000 × 2,700,000/3,000,000 = 900,000      │
    │  ideal_USDC = 1,000,000 × 2,700,000/3,000,000 = 900,000     │
    │  ideal_USDT = 1,000,000 × 2,700,000/3,000,000 = 900,000     │
    │                                                                 │
    │  Step 4: Calculate actual balances (before fees)             │
    │  actual_DAI = 1,000,000 - 300,000 = 700,000                  │
    │  actual_USDC = 1,000,000 (unchanged)                          │
    │  actual_USDT = 1,000,000 (unchanged)                          │
    │                                                                 │
    │  Step 5: Calculate deviations                                 │
    │  deviation_DAI = |700,000 - 900,000| = 200,000               │
    │  deviation_USDC = |1,000,000 - 900,000| = 100,000            │
    │  deviation_USDT = |1,000,000 - 900,000| = 100,000            │
    │                                                                 │
    │  Step 6: Apply fees (0.015% rate)                            │
    │  fee_DAI = 200,000 × 0.015% = 30 DAI                         │
    │  fee_USDC = 100,000 × 0.015% = 15 USDC                       │
    │  fee_USDT = 100,000 × 0.015% = 15 USDT                       │
    │                                                                 │
    │  Step 7: Calculate final amounts                              │
    │  Net withdrawal = Gross - (fees applied to withdrawal token)  │
    │  Net DAI = 300,000 - additional_fee_from_recalculation ≈ 299,900 │
    │  Total fee ≈ 100 DAI equivalent                               │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Fee Visualization:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Fee Impact Analysis                         │
    │                                                                 │
    │   Withdrawal Amount                                            │
    │        │                                                       │
    │        │  ┌─────────────────┐ ← Gross Amount (300,000)        │
    │        │  │                 │                                 │
    │        │  │   User Gets     │ ← Net Amount (~299,900)         │
    │        │  │                 │                                 │
    │        │  ├─────────────────┤                                 │
    │        │  │     Fees        │ ← Trading Fees (~100)           │
    │        │  └─────────────────┘                                 │
    │        │                                                       │
    │        └───────────────────────────────────────► Token Amount │
    │                                                                 │
    │  Fee percentage ≈ 0.033% of withdrawal amount                 │
    │  (Lower than full swap fee due to partial imbalance)          │
    └─────────────────────────────────────────────────────────────────┘
    
    Why Fees Are Applied:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Economic Rationale                         │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Without Fees:                                                 │
    │  ❌ Arbitrageurs could exploit price differences               │
    │  ❌ Pool becomes imbalanced, hurting other LPs                 │
    │  ❌ Free option to exit at ideal price always                 │
    │                                                                 │
    │  With Imbalance Fees:                                         │
    │  ✅ Prevents arbitrage exploitation                           │
    │  ✅ Compensates remaining LPs for increased risk              │
    │  ✅ Maintains pool stability and balance                      │
    │  ✅ Creates fair pricing for convenience                      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Return Values:
    - First return value (dy): Net amount user receives after all fees
    - Second return value (dy_0 - dy): Total fee amount charged
    
    Use Cases:
    1. **Frontend Estimation**: Show users expected withdrawal amounts
    2. **Slippage Calculation**: Compare with user's minimum expectations
    3. **Fee Analysis**: Understand cost of single-token vs balanced withdrawal
    4. **Liquidity Analysis**: Assess impact of large withdrawals on pool
    
    @param _token_amount Number of LP tokens to burn
    @param i Index of token to withdraw (0=DAI, 1=USDC, 2=USDT)
    @return (net_withdrawal_amount, total_fees)
    """
    # Get current pool parameters
    amp: uint256 = self._A()
    
    # Calculate reduced fee rate for single-token withdrawals
    # Same as imbalanced withdrawal: 1/4 of the swap fee
    _fee: uint256 = self.fee * N_COINS / (4 * (N_COINS - 1))
    precisions: uint256[N_COINS] = PRECISION_MUL
    total_supply: uint256 = self.token.totalSupply()

    # Get current normalized balances
    xp: uint256[N_COINS] = self._xp()

    # Calculate current pool invariant D₀
    D0: uint256 = self.get_D(xp, amp)
    
    # Calculate target invariant D₁ after burning LP tokens
    # D₁ = D₀ × (1 - LP_burned / total_LP_supply)
    D1: uint256 = D0 - _token_amount * D0 / total_supply
    
    # Create working copy of current balances for fee calculations
    xp_reduced: uint256[N_COINS] = xp

    # Calculate new balance of withdrawal token to achieve D₁
    # This gives us the gross withdrawal amount (before fees)
    new_y: uint256 = self.get_y_D(amp, i, xp, D1)
    
    # Calculate gross withdrawal amount (before fees)
    # Convert from normalized precision back to token precision
    dy_0: uint256 = (xp[i] - new_y) / precisions[i]

    # Calculate imbalance fees by comparing actual vs ideal balance changes
    for j in range(N_COINS):
        # Calculate expected balance change for each token in ideal case
        dx_expected: uint256 = 0
        
        if j == i:
            # For withdrawal token: expected change = ideal_new_balance - actual_new_balance
            # ideal_new_balance = current_balance × D₁/D₀
            # actual_new_balance = new_y (calculated above)
            dx_expected = xp[j] * D1 / D0 - new_y
        else:
            # For other tokens: expected change = current_balance - ideal_new_balance
            # In ideal case: ideal_new_balance = current_balance × D₁/D₀
            dx_expected = xp[j] - xp[j] * D1 / D0
        
        # Apply fee to the expected deviation and reduce the balance accordingly
        # This simulates the effect of fees on the pool state
        xp_reduced[j] -= _fee * dx_expected / FEE_DENOMINATOR

    # Recalculate withdrawal amount using the fee-adjusted balances
    # This accounts for the fees charged on the imbalance
    dy: uint256 = xp_reduced[i] - self.get_y_D(amp, i, xp_reduced, D1)
    
    # Convert to token precision and subtract 1 wei for rounding safety
    dy = (dy - 1) / precisions[i]

    # Return both the net amount (after fees) and the total fee amount
    return dy, dy_0 - dy


@view
@external
def calc_withdraw_one_coin(_token_amount: uint256, i: int128) -> uint256:
    """
    @notice Calculate the amount of tokens received for single-token withdrawal (public interface)
    @dev This is the public wrapper function for `_calc_withdraw_one_coin()`. It provides
         a simple interface for frontends, aggregators, and users to estimate withdrawal
         amounts before executing the actual transaction.
    
    Function Purpose and Usage:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Public Estimation Interface                  │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  Purpose: Estimate withdrawal amount WITHOUT executing trade    │
    │  Returns: Net amount user will receive (after all fees)        │
    │  Gas Cost: Low (view function, no state changes)               │
    │  Use Cases:                                                     │
    │  - Frontend estimation and display                             │
    │  - Slippage calculation                                        │
    │  - Arbitrage opportunity analysis                              │
    │  - Portfolio rebalancing planning                              │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Practical Example for Frontend Usage:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Frontend Integration                        │
    │                                                                 │
    │  User Input:                                                   │
    │  - LP tokens to burn: 100,000                                 │
    │  - Desired token: USDC (i = 1)                                │
    │                                                                 │
    │  Function Call:                                                │
    │  estimated_usdc = pool.calc_withdraw_one_coin(100000e18, 1)    │
    │  // Returns: 99,950 USDC (example)                            │
    │                                                                 │
    │  Display to User:                                              │
    │  "You will receive approximately 99,950 USDC"                 │
    │  "Fee: ~50 USDC (0.05%)"                                      │
    │  "Click 'Withdraw' to confirm"                                │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Comparison with Other Calculation Functions:
    ┌─────────────────────────────────────────────────────────────────┐
    │                Function Comparison Overview                     │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  calc_withdraw_one_coin():                                     │
    │  ✓ Single token withdrawal estimation                          │
    │  ✓ Includes all fees in calculation                           │
    │  ✓ Returns net amount user receives                           │
    │  ✓ Used for: UX display, slippage protection                  │
    │                                                                 │
    │  calc_token_amount():                                          │
    │  ✓ Multi-token deposit/withdrawal estimation                   │
    │  ✓ Balanced or imbalanced operations                          │
    │  ✓ Returns LP tokens needed/burned                            │
    │  ✓ Used for: Liquidity provision planning                     │
    │                                                                 │
    │  get_dy():                                                     │
    │  ✓ Token swap amount estimation                               │
    │  ✓ Includes swap fees                                         │
    │  ✓ Returns tokens received for token swaps                    │
    │  ✓ Used for: Trading, arbitrage analysis                      │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Economic Interpretation:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Cost-Benefit Analysis                      │
    │                                                                 │
    │  Benefits of Single Token Withdrawal:                         │
    │  ✅ Convenience: Get exactly the token you need               │
    │  ✅ No rebalancing: Don't need to sell unwanted tokens        │
    │  ✅ Strategic: Target specific tokens for opportunities        │
    │                                                                 │
    │  Costs of Single Token Withdrawal:                            │
    │  🟡 Imbalance Fees: ~0.015% of deviation amount               │
    │  🟡 Gas Costs: Slightly higher than balanced withdrawal       │
    │  🟡 Slippage: Creates pool imbalance                          │
    │                                                                 │
    │  When to Use Single Token Withdrawal:                         │
    │  💰 Need specific token for payments/opportunities            │
    │  💰 Fee cost < convenience benefit                            │
    │  💰 Small withdrawals (lower fee impact)                      │
    │  💰 Pool is already imbalanced in your favor                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Slippage Protection Integration:
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Slippage Calculation Guide                    │
    │                                                                 │
    │  Step 1: Get current estimate                                  │
    │  current_estimate = calc_withdraw_one_coin(lp_amount, i)       │
    │                                                                 │
    │  Step 2: Apply slippage tolerance (e.g., 0.5%)               │
    │  slippage_tolerance = 0.005  // 0.5%                          │
    │  min_amount = current_estimate * (1 - slippage_tolerance)      │
    │                                                                 │
    │  Step 3: Use in actual withdrawal                             │
    │  pool.remove_liquidity_one_coin(lp_amount, i, min_amount)     │
    │                                                                 │
    │  Example:                                                      │
    │  - Estimate: 100,000 USDC                                     │
    │  - Tolerance: 0.5%                                            │
    │  - Min amount: 99,500 USDC                                    │
    │  - Protection: Transaction reverts if < 99,500 USDC           │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Gas Optimization Notes:
    - This is a view function (no gas cost when called externally)
    - Internal calculations are optimized for gas efficiency
    - Consider caching results if calling multiple times
    - Batch with other view calls for better efficiency
    
    Error Scenarios:
    - Invalid token index (i): Function will revert
    - Zero LP token amount: Returns 0
    - Pool killed: Function still works (view only)
    - Insufficient pool balance: May return unrealistic values
    
    @param _token_amount Number of LP tokens to burn for withdrawal estimation
    @param i Index of token to receive (0=DAI, 1=USDC, 2=USDT)
    @return Net amount of tokens user will receive after all fees
    """
    return self._calc_withdraw_one_coin(_token_amount, i)[0]


@external
@nonreentrant('lock')
def remove_liquidity_one_coin(_token_amount: uint256, i: int128, min_amount: uint256):
    """
    @notice Remove liquidity by burning LP tokens and receiving a single token type
    @dev This function implements single-token withdrawal, allowing users to burn their
         LP tokens and receive only one type of underlying asset. This creates pool
         imbalance and incurs fees, but provides convenience and targeted exposure.
    
    Single Token Withdrawal Complete Process:
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Complete Withdrawal Flow                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  1. 🔒 Security Checks                                          │
    │     ✓ Pool not killed/emergency shutdown                       │
    │     ✓ Reentrancy protection (nonreentrant lock)               │
    │     ✓ Valid token index and amounts                           │
    │                                                                 │
    │  2. 📊 Calculate Withdrawal Amount                              │
    │     ✓ Call _calc_withdraw_one_coin() for precise calculation   │
    │     ✓ Apply imbalance fees to withdrawal amount                │
    │     ✓ Calculate admin fee portion                              │
    │                                                                 │
    │  3. 🛡️ Slippage Protection                                      │
    │     ✓ Check withdrawal amount >= min_amount                    │
    │     ✓ Prevent excessive slippage due to market changes         │
    │                                                                 │
    │  4. 💰 Execute Financial Operations                             │
    │     ✓ Update pool balance (subtract withdrawn + admin fees)    │
    │     ✓ Burn LP tokens from user's account                      │
    │     ✓ Transfer tokens to user safely                          │
    │                                                                 │
    │  5. 📝 Record Transaction                                       │
    │     ✓ Emit RemoveLiquidityOne event                           │
    │     ✓ Log for off-chain tracking and analytics                │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Mathematical Process Overview:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Math Behind the Scenes                     │
    │                                                                 │
    │  Before Withdrawal:                                            │
    │  Pool: [x₀, x₁, x₂] = [1000k DAI, 1000k USDC, 1000k USDT]    │
    │  D₀ = 3,000,000 (invariant)                                   │
    │  LP_supply = 2,950,000                                        │
    │                                                                 │
    │  User Action: Burn 295k LP for DAI only                       │
    │                                                                 │
    │  Step 1: Calculate target invariant                           │
    │  D₁ = D₀ × (1 - LP_burned/LP_total)                          │
    │     = 3,000,000 × (1 - 295,000/2,950,000) = 2,700,000        │
    │                                                                 │
    │  Step 2: Solve for new DAI balance with invariant D₁          │
    │  Using get_y_D(): new_DAI_balance ≈ 700,000                   │
    │  Gross withdrawal = 1,000,000 - 700,000 = 300,000 DAI        │
    │                                                                 │
    │  Step 3: Apply imbalance fees                                 │
    │  Fee rate = 0.04% × 3/(4×2) = 0.015%                         │
    │  Net withdrawal ≈ 299,900 DAI                                 │
    │  Admin fees ≈ 25 DAI (50% of 50 DAI total fees)              │
    │                                                                 │
    │  After Withdrawal:                                             │
    │  Pool: [700,075 DAI, 1000k USDC, 1000k USDT]                 │
    │  LP_supply = 2,655,000                                        │
    │  User receives: 299,900 DAI                                   │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Visual Representation of Pool Changes:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Pool State Transition                       │
    │                                                                 │
    │  Before:                          After:                       │
    │  ┌─────────────────────┐         ┌─────────────────────┐       │
    │  │  Pool Composition   │         │  Pool Composition   │       │
    │  │                     │         │                     │       │
    │  │  🟡 DAI:   1000k    │   →     │  🟡 DAI:   700k     │ ← -30%│
    │  │  🔵 USDC:  1000k    │         │  🔵 USDC:  1000k    │ ← Same│
    │  │  🟢 USDT:  1000k    │         │  🟢 USDT:  1000k    │ ← Same│
    │  │                     │         │                     │       │
    │  │  Balance: Ideal     │         │  Balance: Skewed    │       │
    │  │  LP Supply: 2950k   │         │  LP Supply: 2655k   │       │
    │  └─────────────────────┘         └─────────────────────┘       │
    │                                                                 │
    │  Pool becomes imbalanced toward USDC/USDT                     │
    │  Creates arbitrage opportunities for rebalancing              │
    └─────────────────────────────────────────────────────────────────┘
    
    Fee Structure and Economic Impact:
    ┌─────────────────────────────────────────────────────────────────┐
    │                       Fee Breakdown                            │
    │                                                                 │
    │  Base Pool Fee: 0.04% (for regular swaps)                     │
    │  Imbalance Fee: 0.04% × 3/(4×2) = 0.015%                      │
    │                                                                 │
    │  Why Lower Fee for Withdrawals?                                │
    │  • Encourages liquidity provision                             │
    │  • Less harmful than pure arbitrage swaps                     │
    │  • Partially offsets by requiring pool rebalancing            │
    │                                                                 │
    │  Fee Distribution:                                             │
    │  ┌─────────────────────────────────────────────────────────── │
    │  │              Total Fees (100 DAI)                         │ │
    │  ├─────────────────────────────────────────────────────────── │
    │  │  Admin Fee (50%)     │  LP Fee (50%)                     │ │
    │  │  → Protocol Treasury │  → Remaining LPs                  │ │
    │  │  → 50 DAI           │  → 50 DAI                         │ │
    │  └─────────────────────────────────────────────────────────── │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Security Features and Protections:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      Security Measures                         │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  🔒 Reentrancy Protection:                                     │
    │     • @nonreentrant('lock') decorator                          │
    │     • Prevents recursive calls during execution               │
    │     • Protects against flash loan attacks                     │
    │                                                                 │
    │  🛡️ Emergency Protection:                                      │
    │     • Pool kill switch (is_killed check)                      │
    │     • Prevents operations during emergencies                  │
    │     • Admin can halt all withdrawals if needed                │
    │                                                                 │
    │  💸 Slippage Protection:                                       │
    │     • min_amount parameter enforcement                         │
    │     • Transaction reverts if amount < minimum                  │
    │     • Prevents sandwich attacks and MEV exploitation          │
    │                                                                 │
    │  🔐 Safe Token Transfers:                                      │
    │     • Custom ERC20 transfer implementation                     │
    │     • Handles tokens that don't return bool                   │
    │     • Validates transfer success explicitly                    │
    │                                                                 │
    │  ⚖️ Precise Accounting:                                        │
    │     • Exact balance tracking                                   │
    │     • Prevents rounding exploits                              │
    │     • Admin fee calculation safeguards                        │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Use Cases and Strategic Applications:
    ┌─────────────────────────────────────────────────────────────────┐
    │                      When to Use This Function                 │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  💼 Portfolio Management:                                      │
    │     • Reduce exposure to specific stablecoins                 │
    │     • Rebalance portfolio allocations                         │
    │     • Take profits in preferred currency                      │
    │                                                                 │
    │  🏦 Operational Needs:                                         │
    │     • Get specific token for loan repayments                  │
    │     • Meet margin requirements in particular asset            │
    │     • Cover expenses in required currency                     │
    │                                                                 │
    │  📈 Trading Opportunities:                                     │
    │     • Exit to specific token for external opportunities       │
    │     • Arbitrage with other protocols/DEXs                     │
    │     • Prepare for known market events                         │
    │                                                                 │
    │  ⚡ Convenience Factors:                                       │
    │     • Avoid multiple token management                          │
    │     • Reduce gas costs from token conversions                 │
    │     • Simplify tax accounting                                  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Comparison with Other Withdrawal Methods:
    ┌─────────────────────────────────────────────────────────────────┐
    │                  Withdrawal Method Comparison                   │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │  remove_liquidity() - Balanced Withdrawal:                    │
    │  ✅ Fee: 0% (no trading fees)                                  │
    │  ✅ Speed: Fast execution                                      │
    │  ❌ Output: Get all 3 tokens proportionally                   │
    │  ❌ Use case: When you want all tokens                        │
    │                                                                 │
    │  remove_liquidity_imbalance() - Custom Amounts:               │
    │  🟡 Fee: 0.015% on imbalanced portions                        │
    │  🟡 Speed: Moderate (more complex calculations)               │
    │  ✅ Output: Exact amounts of each token you specify           │
    │  ✅ Use case: Precise portfolio rebalancing                   │
    │                                                                 │
    │  remove_liquidity_one_coin() - Single Token:                  │
    │  🔴 Fee: 0.015% on full withdrawal amount                     │
    │  🔴 Speed: Moderate (complex fee calculations)                │
    │  ✅ Output: Only the token you want                           │
    │  ✅ Use case: Maximum convenience, targeted exposure          │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    
    Event Logging and Analytics:
    The function emits `RemoveLiquidityOne(provider, token_amount, coin_amount)`
    This enables:
    - Historical withdrawal tracking
    - Volume analytics for each token
    - User behavior analysis
    - DeFi protocol integrations
    - Tax reporting assistance
    
    @param _token_amount Number of LP tokens to burn for withdrawal
    @param i Index of token to receive (0=DAI, 1=USDC, 2=USDT)
    @param min_amount Minimum amount of tokens to receive (slippage protection)
    """
    # Security check: Ensure pool is not in emergency shutdown state
    assert not self.is_killed  # dev: is killed

    # Calculate both the net withdrawal amount and fees using internal function
    dy: uint256 = 0
    dy_fee: uint256 = 0
    dy, dy_fee = self._calc_withdraw_one_coin(_token_amount, i)
    
    # Slippage protection: Ensure user gets at least their minimum expected amount
    # This prevents losses due to:
    # - Market movements between estimation and execution
    # - MEV attacks and sandwich attacks
    # - Unexpected pool state changes
    assert dy >= min_amount, "Not enough coins removed"

    # Update pool balance accounting:
    # - Subtract the amount withdrawn (dy)
    # - Subtract the admin fee portion (admin gets percentage of total fees)
    # - The remaining fee stays in the pool to benefit other LPs
    self.balances[i] -= (dy + dy_fee * self.admin_fee / FEE_DENOMINATOR)
    
    # Burn LP tokens from user's account
    # This permanently reduces the total LP token supply
    # If user doesn't have enough LP tokens, this will revert
    self.token.burnFrom(msg.sender, _token_amount)  # dev: insufficient funds

    # Safe token transfer to user
    # Custom implementation that works with both:
    # - Standard ERC20 tokens that return bool
    # - Non-standard tokens (like USDT) that don't return values
    _response: Bytes[32] = raw_call(
        self.coins[i],
        concat(
            method_id("transfer(address,uint256)"),
            convert(msg.sender, bytes32),
            convert(dy, bytes32),
        ),
        max_outsize=32,
    )  # dev: failed transfer
    
    # Verify transfer succeeded for tokens that return bool
    if len(_response) > 0:
        assert convert(_response, bool)  # dev: failed transfer

    # Emit event for off-chain tracking and analytics
    # Parameters: user_address, lp_tokens_burned, tokens_received
    log RemoveLiquidityOne(msg.sender, _token_amount, dy)


### Admin functions # allows you to change the amplification factor
@external
def ramp_A(_future_A: uint256, _future_time: uint256):
    """
    @notice Gradually adjusts the amplification coefficient A over time
    @dev The amplification parameter affects the StableSwap invariant curve:
         
         Visual representation of curve shapes with different A values:
                     
         Token Balance Y
         │
         │    ┌─────── A = ∞ (straight line, constant sum)
         │   ╱│
         │  ╱ │
         │ ╱  │        A = 200 (high)
         │╱   │
         ┼─────────── A = 50 (medium)
         │╲   │
         │ ╲  │
         │  ╲ │        A = 5 (low)
         │   ╲│
         │    └─────── A = 1 (hyperbola, constant product)
         │
         └───────────────────────── Token Balance X
         
         - Higher A = More stable coin behavior (closer to constant sum)
           Example: A=200 means 0.5% slippage when pool is imbalanced 5:1
           
         - Lower A = More AMM-like behavior (closer to constant product)
           Example: A=5 means 10% slippage when pool is imbalanced 5:1
         
         A is gradually changed to prevent flash-loan attacks/manipulation.
         
    @param _future_A The target A value (must be between 0 and MAX_A)
    @param _future_time Timestamp when A should reach the target value
    """
    # Only the contract owner can call this function
    assert msg.sender == self.owner  # dev: only owner
    
    # Safety check: Ensure at least MIN_RAMP_TIME (1 day) has passed since the last A change
    # This prevents changing A too frequently, which could be exploited
    assert block.timestamp >= self.initial_A_time + MIN_RAMP_TIME # make sure that we are not ramping too fast
    
    # Safety check: Ensure the future time is at least MIN_RAMP_TIME (1 day) in the future
    # This ensures the change happens gradually, not instantly
    assert _future_time >= block.timestamp + MIN_RAMP_TIME  # dev: insufficient time

    # Get the current A value by calling the internal _A() function
    # The _A() function handles the linear interpolation between initial_A and future_A
    _initial_A: uint256 = self._A()

    # Check that future A is in the valid range: greater than 0 and less than MAX_A (10^6)
    # A=0 would break the pool, and very high A values could lead to integer overflows
    assert (_future_A > 0) and (_future_A < MAX_A)
    
    # Check that the change in A is not too drastic to prevent manipulation
    # Example: If MAX_A_CHANGE = 10:
    #   - If increasing A: If current A = 100, max new A = 1000
    #   - If decreasing A: If current A = 100, min new A = 10
    # This prevents rapid changes that could harm liquidity providers
    assert ((_future_A >= _initial_A) and (_future_A <= _initial_A * MAX_A_CHANGE)) or
           ((_future_A < _initial_A) and (_future_A * MAX_A_CHANGE >= _initial_A))
    
    # Store the current A as the starting point for the ramp
    self.initial_A = _initial_A
    # Store the target A value
    self.future_A = _future_A
    # Store the current timestamp as the starting time for the ramp
    self.initial_A_time = block.timestamp
    # Store the future timestamp when the ramp will be complete
    self.future_A_time = _future_time

    # Example of linear interpolation during ramp period:
    # If initial_A = 100, future_A = 200, initial_time = now, future_time = now + 7 days
    # After 0 days: A = 100
    # After 1 day:  A = 100 + (200-100)*(1/7) = 114.3
    # After 3 days: A = 100 + (200-100)*(3/7) = 142.9
    # After 7 days: A = 200

    # Emit an event to log this change
    log RampA(_initial_A, _future_A, block.timestamp, _future_time)


# Allows to stop ramping A and sets the current A as both initial and future A
@external
def stop_ramp_A():
    """
    @notice Stops the gradual change of the amplification parameter
    @dev This function can be called by the owner if they want to halt 
         the ramping of A and set it to the current value.
         
         Example use cases:
         1. Emergency: If a ramping A is causing unexpected behavior
         2. Optimization: If the current A value is optimal for the pool
         3. Market conditions: If conditions change and the target A is no longer suitable
         
         Visual representation of stopping a ramp:
         
         Amplification (A)
         │
         │               Target A (never reached)
         │                    x
         │                   ╱
         │                  ╱
         │                 ╱
         │                ╱
         │               ╱ ramp halted
         │              *────────────► Current A (new permanent A)
         │             ╱
         │            ╱
         │           ╱
         │          ╱
         │         ╱
         │        ╱
         │       ╱
         │ Initial A
         │
         └───────────────────────────────────── Time
    """
    # Only the contract owner can call this function
    assert msg.sender == self.owner  # dev: only owner

    # Get the current interpolated A value
    current_A: uint256 = self._A()
    
    # Stop the ramp by setting both initial and future A to the current value
    self.initial_A = current_A
    self.future_A = current_A
    
    # Set both timestamps to the current time, which effectively
    # prevents _A() from doing any interpolation in the future
    self.initial_A_time = block.timestamp
    self.future_A_time = block.timestamp
    
    # Since future_A_time is set to the present, _A() will always
    # use the else clause and return the fixed current_A value

    # Emit an event to log this change
    log StopRampA(current_A, block.timestamp)


@external
def commit_new_fee(new_fee: uint256, new_admin_fee: uint256):
    assert msg.sender == self.owner  # dev: only owner
    assert self.admin_actions_deadline == 0  # dev: active action
    assert new_fee <= MAX_FEE  # dev: fee exceeds maximum
    assert new_admin_fee <= MAX_ADMIN_FEE  # dev: admin fee exceeds maximum

    _deadline: uint256 = block.timestamp + ADMIN_ACTIONS_DELAY
    self.admin_actions_deadline = _deadline
    self.future_fee = new_fee
    self.future_admin_fee = new_admin_fee

    log CommitNewFee(_deadline, new_fee, new_admin_fee)


@external
def apply_new_fee():
    assert msg.sender == self.owner  # dev: only owner
    assert block.timestamp >= self.admin_actions_deadline  # dev: insufficient time
    assert self.admin_actions_deadline != 0  # dev: no active action

    self.admin_actions_deadline = 0
    _fee: uint256 = self.future_fee
    _admin_fee: uint256 = self.future_admin_fee
    self.fee = _fee
    self.admin_fee = _admin_fee

    log NewFee(_fee, _admin_fee)

# to revert old modifications of the fee and admin fee
@external
def revert_new_parameters():
    assert msg.sender == self.owner  # dev: only owner

    self.admin_actions_deadline = 0

# to transfer ownership of the contract to another address
@external
def commit_transfer_ownership(_owner: address):
    assert msg.sender == self.owner  # dev: only owner
    assert self.transfer_ownership_deadline == 0  # dev: active transfer

    _deadline: uint256 = block.timestamp + ADMIN_ACTIONS_DELAY
    self.transfer_ownership_deadline = _deadline
    self.future_owner = _owner

    log CommitNewAdmin(_deadline, _owner)

# to apply the transfer of ownership to the new owner
@external
def apply_transfer_ownership():
    assert msg.sender == self.owner  # dev: only owner
    assert block.timestamp >= self.transfer_ownership_deadline  # dev: insufficient time
    assert self.transfer_ownership_deadline != 0  # dev: no active transfer

    self.transfer_ownership_deadline = 0
    _owner: address = self.future_  owner
    self.owner = _owner

    log NewAdmin(_owner)


@external
def revert_transfer_ownership():
    assert msg.sender == self.owner  # dev: only owner

    self.transfer_ownership_deadline = 0


@view
# self.balances[i] represents the tokens that "belong" to users (liquidity providers, traders, etc.) 
# ERC20(self.coins[i]).balanceOf(self) actually represents the total balance of the contract in that coin
# so the difference between the two gives the admin's balance in that coin
@external
def admin_balances(i: uint256) -> uint256:
    return ERC20(self.coins[i]).balanceOf(self) - self.balances[i]

# @note transfers all admin fees to the owner of the contract
@external
def withdraw_admin_fees():
    assert msg.sender == self.owner  # dev: only owner

    for i in range(N_COINS):
        c: address = self.coins[i]
        value: uint256 = ERC20(c).balanceOf(self) - self.balances[i]
        if value > 0:
            # "safeTransfer" which works for ERC20s which return bool or not
            _response: Bytes[32] = raw_call(
                c,
                concat(
                    method_id("transfer(address,uint256)"),
                    convert(msg.sender, bytes32),
                    convert(value, bytes32),
                ),
                max_outsize=32,
            )  # dev: failed transfer
            if len(_response) > 0:
                assert convert(_response, bool)  # dev: failed transfer

# This function allows the contract owner to "donate" or "forfeit" all accumulated admin fees back to the pool users.
# allows an admin to return the admin fees to the pool, effectively redistributing them among the liquidity providers.
# new D will get caculated when you perform any action on the pool like adding liquidity or removing liquidity or exchanging tokens
@external
def donate_admin_fees():
    assert msg.sender == self.owner  # dev: only owner
    for i in range(N_COINS):
        self.balances[i] = ERC20(self.coins[i]).balanceOf(self)

#@note revokes operation such as add_liquidity, remove_liquidity, exchange, etc.
@external
def kill_me():
    assert msg.sender == self.owner  # dev: only owner
    assert self.kill_deadline > block.timestamp  # dev: deadline has passed
    self.is_killed = True

# note this function is used to unkill the pool so you can perform operations on it again
@external
def unkill_me():
    assert msg.sender == self.owner  # dev: only owner
    self.is_killed = False
