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
    @notice Contract constructor
    @param _owner Contract owner address
    @param _coins Addresses of ERC20 conracts of coins
    @param _pool_token Address of the token representing LP share
    @param _A Amplification coefficient multiplied by n * (n - 1)
    @param _fee Fee to charge for exchanges
    @param _admin_fee Admin fee
    """
    for i in range(N_COINS):
        assert _coins[i] != ZERO_ADDRESS
    self.coins = _coins
    self.initial_A = _A
    self.future_A = _A
    self.fee = _fee
    self.admin_fee = _admin_fee
    self.owner = _owner
    self.kill_deadline = block.timestamp + KILL_DEADLINE_DT
    self.token = CurveToken(_pool_token)


@view
@internal
def _A() -> uint256: #Amplication factor , which gradually changes from initial_A to future_A and the future_A is set by the owner 
    """
    Handle ramping A up or down
    """
    t1: uint256 = self.future_A_time
    A1: uint256 = self.future_A

    if block.timestamp < t1:
        A0: uint256 = self.initial_A
        t0: uint256 = self.initial_A_time
        # Expressions in uint256 cannot have negative numbers, thus "if"
        if A1 > A0:
            # @note
            # At t0 : it all goes to A0 
            # At t1 : it all goes to A1 

            # at T0 (block.timestamp - t0) / (t1 - t0) ==0 so it will return A0
            # at T1 (block.timestamp - t0) / (t1 - t0) ==1 so it will return A1 After A0 + A1 - A0
            return A0 + (A1 - A0) * (block.timestamp - t0) / (t1 - t0)
        else:
            # @note
            # Same thing but when A1 < A0 so th Amplication factor is decreasing
            # at T0 (block.timestamp - t0) / (t1 - t0) ==0 so it will return A0
            # at T1 (block.timestamp - t0) / (t1 - t0) ==1 so it will return A1 After A0 - (A0 - A1)

            return A0 - (A0 - A1) * (block.timestamp - t0) / (t1 - t0)

    else:  # when t1 == 0 or block.timestamp >= t1
        return A1
 

@view
@external
def A() -> uint256:
    return self._A()


@view
@internal
# make sure everything is in the same precision adn decimals
def _xp() -> uint256[N_COINS]:
    result: uint256[N_COINS] = RATES
    for i in range(N_COINS):
        # making sure all have same decimals
        # helps in converting to the same precision and decimals
        # DAI = 1e18 * dai Balance / 1e18 = 18 decimals 
        # USDC = 1e30 * usdc Balance / 1e18 = 18 decimals 
        # USDT = 1e30 * usdt Balance / 1e18 = 18 decimals
        # @note even though USDC and USDT have 6 decimals, we multiply by 1e30 to make sure they are in the same precision
        result[i] = result[i] * self.balances[i] / LENDING_PRECISION
    return result


@pure
@internal #note gives xp which is the normalized balances of the coins in the pool with accurate precision pertaining to its decimals but do not change the state of the pool uses the stored balances of the pool 
def _xp_mem(_balances: uint256[N_COINS]) -> uint256[N_COINS]:
    result: uint256[N_COINS] = RATES
    for i in range(N_COINS):
        result[i] = result[i] * _balances[i] / PRECISION
    return result


@pure
@internal #xp represents the normalized balances of the coins in the pool with accurate precision
# performs Newton's method to find the invariant D

# @note D parameter represents the total liquidity in the pool 
def get_D(xp: uint256[N_COINS], amp: uint256) -> uint256:
    S: uint256 = 0
    for _x in xp:
        S += _x
    if S == 0:
        return 0

    Dprev: uint256 = 0
    D: uint256 = S
    Ann: uint256 = amp * N_COINS

    # performs newton's method to find the invariant D
    for _i in range(255):
        D_P: uint256 = D
        for _x in xp:
            D_P = D_P * D / (_x * N_COINS)  # If division by 0, this will be borked: only withdrawal will work. And that is good
        Dprev = D
        D = (Ann * S + D_P * N_COINS) * D / ((Ann - 1) * D + (N_COINS + 1) * D_P)
        # Equality with the precision of 1
        # found a solution 
        if D > Dprev:
            if D - Dprev <= 1:
                break
        else:
            if Dprev - D <= 1:
                break
    return D


@view
@internal # @note this function is used to get The D value without actually changing the balances of the pool and chaning the state of the pool it gives you the D value based on the balances Stored in it . 
def get_D_mem(_balances: uint256[N_COINS], amp: uint256) -> uint256:
    return self.get_D(self._xp_mem(_balances), amp)


@view
@external
# @note represent Value of each LP token in the pool
def get_virtual_price() -> uint256:
    """
    Returns portfolio virtual price (for calculating profit)
    scaled up by 1e18
    """
    D: uint256 = self.get_D(self._xp(), self._A())
    # D is in the units similar to DAI (e.g. converted to precision 1e18)
    # When balanced, D = n * x_u - total virtual value of the portfolio
    token_supply: uint256 = self.token.totalSupply() # total supply of the LP token that were minted
    return D * PRECISION / token_supply # value of each lp token


@view
@external
# @note it returns the amount of lp token to be burned or minted
def calc_token_amount(amounts: uint256[N_COINS], deposit: bool) -> uint256:
    """
    Simplified method to calculate addition or reduction in token supply at
    deposit or withdrawal without taking fees into account (but looking at
    slippage).
    Needed to prevent front-running, not for precise calculations!
    """
    # Gets the balances of the pool
    _balances: uint256[N_COINS] = self.balances
    amp: uint256 = self._A()
    D0: uint256 = self.get_D_mem(_balances, amp) #
    for i in range(N_COINS):
        if deposit:
            _balances[i] += amounts[i]
        else:
            _balances[i] -= amounts[i]
    # calculate the D value based on the current/updated balances of the pool 
    D1: uint256 = self.get_D_mem(_balances, amp) #get the D1 Parameter after adding or removing the coins from the pool 
    token_amount: uint256 = self.token.totalSupply()
    diff: uint256 = 0
    if deposit:
        diff = D1 - D0
    else:
        diff = D0 - D1

    return diff * token_amount / D0

@external
@nonreentrant('lock')
def add_liquidity(amounts: uint256[N_COINS], min_mint_amount: uint256):
    assert not self.is_killed  # dev: is killed
    # this fees array is used if you are going to make the pool imbalance
    fees: uint256[N_COINS] = empty(uint256[N_COINS])
    # calculating the fees on imbalance
    # x -> Add liquidity
    # remove liquidity -> y 

    # in curve you can add liqudiity in any coin you want and withdraw in any coin you want so this way you can perform swaps by not doing any swap fees so we also have a fees for that up here
    _fee: uint256 = self.fee * N_COINS / (4 * (N_COINS - 1))
    _admin_fee: uint256 = self.admin_fee
    amp: uint256 = self._A()

    token_supply: uint256 = self.token.totalSupply()
    # Initial invariant
    D0: uint256 = 0
    old_balances: uint256[N_COINS] = self.balances
    if token_supply > 0:
        D0 = self.get_D_mem(old_balances, amp)
    new_balances: uint256[N_COINS] = old_balances

    for i in range(N_COINS):
        in_amount: uint256 = amounts[i]
        if token_supply == 0:
            assert in_amount > 0  # dev: initial deposit requires all coins
        in_coin: address = self.coins[i]

        # Take coins from the sender
        if in_amount > 0:
            if i == FEE_INDEX:
                in_amount = ERC20(in_coin).balanceOf(self)

            # "safeTransferFrom" which works for ERC20s which return bool or not
            # usdt does not return bool on transferFrom
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
            if len(_response) > 0:
                assert convert(_response, bool)  # dev: failed transfer
            # no fees index 
            if i == FEE_INDEX:
                in_amount = ERC20(in_coin).balanceOf(self) - in_amount

        new_balances[i] = old_balances[i] + in_amount

    # Invariant after change
    D1: uint256 = self.get_D_mem(new_balances, amp)
    assert D1 > D0

    # We need to recalculate the invariant accounting for fees
    # to calculate fair user's share
    D2: uint256 = D1
    if token_supply > 0:
        # Only account for fees if we are not the first to deposit
        for i in range(N_COINS):
            ideal_balance: uint256 = D1 * old_balances[i] / D0
            difference: uint256 = 0
            if ideal_balance > new_balances[i]:
                difference = ideal_balance - new_balances[i]
            else:
                difference = new_balances[i] - ideal_balance
            # imbalance fee on each coin
            fees[i] = _fee * difference / FEE_DENOMINATOR
            # minus the fees from the balances
            self.balances[i] = new_balances[i] - (fees[i] * _admin_fee / FEE_DENOMINATOR)
            new_balances[i] -= fees[i]
        D2 = self.get_D_mem(new_balances, amp)
    else:
        self.balances = new_balances

    # Calculate, how much pool tokens to mint
    mint_amount: uint256 = 0
    if token_supply == 0:
        mint_amount = D1  # Take the dust if there was any
    else:
        mint_amount = token_supply * (D2 - D0) / D0

    assert mint_amount >= min_mint_amount, "Slippage screwed you"

    # Mint pool tokens
    self.token.mint(msg.sender, mint_amount)

    log AddLiquidity(msg.sender, amounts, fees, D1, token_supply + mint_amount)


@view
@internal # computing the output amount of a token when given an input amount of another token
# i is the index of the coin to sell, j is the index of the coin to buy 
# x : the amount of the coin to sell in c-units
# xp : nomalization of the balances
def get_y(i: int128, j: int128, x: uint256, xp_: uint256[N_COINS]) -> uint256: # new balance of the token out
    # x in the input is converted to the same price/precision

    assert i != j       # dev: same coin
    assert j >= 0       # dev: j below zero
    assert j < N_COINS  # dev: j above N_COINS

    # should be unreachable, but good for safety
    assert i >= 0
    assert i < N_COINS

    amp: uint256 = self._A()
    D: uint256 = self.get_D(xp_, amp)
    c: uint256 = D
    S_: uint256 = 0
    Ann: uint256 = amp * N_COINS
# does some liquidity calculations using the newton's method 
    _x: uint256 = 0
    for _i in range(N_COINS):
        if _i == i:
            _x = x
        elif _i != j:
            _x = xp_[_i]
        else:
            continue
        S_ += _x
        c = c * D / (_x * N_COINS)
    c = c * D / (Ann * N_COINS)
    b: uint256 = S_ + D / Ann  # - D
    y_prev: uint256 = 0
    y: uint256 = D
    for _i in range(255):
        y_prev = y
        y = (y*y + c) / (2 * y + b - D)
        # Equality with the precision of 1
        if y > y_prev:
            if y - y_prev <= 1:
                break
        else:
            if y_prev - y <= 1:
                break
    return y


@view
@external #note get_dy dy   
def get_dy(i: int128, j: int128, dx: uint256) -> uint256:
    # dx and dy in c-units
    rates: uint256[N_COINS] = RATES
    xp: uint256[N_COINS] = self._xp()

    x: uint256 = xp[i] + (dx * rates[i] / PRECISION)
    y: uint256 = self.get_y(i, j, x, xp)
    dy: uint256 = (xp[j] - y - 1) * PRECISION / rates[j]
    _fee: uint256 = self.fee * dy / FEE_DENOMINATOR
    return dy - _fee


@view
@external #this function is used for the yeild bearing assets 
def get_dy_underlying(i: int128, j: int128, dx: uint256) -> uint256:
    # dx and dy in underlying units
    xp: uint256[N_COINS] = self._xp()
    precisions: uint256[N_COINS] = PRECISION_MUL

    x: uint256 = xp[i] + dx * precisions[i]
    y: uint256 = self.get_y(i, j, x, xp)
    dy: uint256 = (xp[j] - y - 1) / precisions[j]
    _fee: uint256 = self.fee * dy / FEE_DENOMINATOR
    return dy - _fee



@external
@nonreentrant('lock')
# i is the index of the coin to sell, j is the index of the coin to buy
# dx is the amount of the coin to sell in c-units
# min_dy is the minimum amount of the coin to buy in c-units
def exchange(i: int128, j: int128, dx: uint256, min_dy: uint256):
    assert not self.is_killed  # dev: is killed
    rates: uint256[N_COINS] = RATES 

    old_balances: uint256[N_COINS] = self.balances # get the old balances of the coins in the pool
    xp: uint256[N_COINS] = self._xp_mem(old_balances) # get the normalized balances of the coins in the pool with accurate precision

    # Handling an unexpected charge of a fee on transfer (USDT, PAXG)
    dx_w_fee: uint256 = dx
    input_coin: address = self.coins[i]

    if i == FEE_INDEX:
        dx_w_fee = ERC20(input_coin).balanceOf(self)

    # "safeTransferFrom" which works for ERC20s which return bool or not
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
    if len(_response) > 0:
        assert convert(_response, bool)  # dev: failed transfer

    if i == FEE_INDEX:
        ## Calculating the actual amount that came in 
        dx_w_fee = ERC20(input_coin).balanceOf(self) - dx_w_fee
    # Scaling 
    x: uint256 = xp[i] + dx_w_fee * rates[i] / PRECISION
    y: uint256 = self.get_y(i, j, x, xp)

    dy: uint256 = xp[j] - y - 1  # -1 just in case there were some rounding errors

    # Apply fee on token out #n @note
    dy_fee: uint256 = dy * self.fee / FEE_DENOMINATOR

    # Convert all to real units 
    dy = (dy - dy_fee) * PRECISION / rates[j]
    assert dy >= min_dy, "Exchange resulted in fewer coins than expected"

    dy_admin_fee: uint256 = dy_fee * self.admin_fee / FEE_DENOMINATOR
    dy_admin_fee = dy_admin_fee * PRECISION / rates[j]

    # Change balances exactly in same way as we change actual ERC20 coin amounts
    self.balances[i] = old_balances[i] + dx_w_fee
    # not including the fees for the admin
    # When rounding errors happen, we undercharge admin fee in favor of LP
    self.balances[j] = old_balances[j] - dy - dy_admin_fee

    # "safeTransfer" which works for ERC20s which return bool or not
    _response = raw_call(
        self.coins[j],
        concat(
            method_id("transfer(address,uint256)"),
            convert(msg.sender, bytes32),
            convert(dy, bytes32),
        ),
        max_outsize=32,
    )  # dev: failed transfer
    if len(_response) > 0:
        assert convert(_response, bool)  # dev: failed transfer

    log TokenExchange(msg.sender, i, dx, j, dy)


@external #you get a balanced token amounts out based on your lp share
@nonreentrant('lock')
def remove_liquidity(_amount: uint256, min_amounts: uint256[N_COINS]):
    total_supply: uint256 = self.token.totalSupply()
    amounts: uint256[N_COINS] = empty(uint256[N_COINS])
    fees: uint256[N_COINS] = empty(uint256[N_COINS])  # Fees are unused but we've got them historically in event

    for i in range(N_COINS):
        value: uint256 = self.balances[i] * _amount / total_supply
        assert value >= min_amounts[i], "Withdrawal resulted in fewer coins than expected"
        self.balances[i] -= value
        amounts[i] = value

        # "safeTransfer" which works for ERC20s which return bool or not
        _response: Bytes[32] = raw_call(
            self.coins[i],
            concat(
                method_id("transfer(address,uint256)"),
                convert(msg.sender, bytes32),
                convert(value, bytes32),
            ),
            max_outsize=32,
        )  # dev: failed transfer
        if len(_response) > 0:
            assert convert(_response, bool)  # dev: failed transfer

    self.token.burnFrom(msg.sender, _amount)  # dev: insufficient funds

    log RemoveLiquidity(msg.sender, amounts, fees, total_supply - _amount)


@external
@nonreentrant('lock')
def remove_liquidity_imbalance(amounts: uint256[N_COINS], max_burn_amount: uint256):
    assert not self.is_killed  # dev: is killed

    token_supply: uint256 = self.token.totalSupply()
    assert token_supply != 0  # dev: zero total supply
    _fee: uint256 = self.fee * N_COINS / (4 * (N_COINS - 1))
    _admin_fee: uint256 = self.admin_fee
    amp: uint256 = self._A()

    old_balances: uint256[N_COINS] = self.balances
    new_balances: uint256[N_COINS] = old_balances
    D0: uint256 = self.get_D_mem(old_balances, amp)
    for i in range(N_COINS):
        new_balances[i] -= amounts[i]
    D1: uint256 = self.get_D_mem(new_balances, amp)
    fees: uint256[N_COINS] = empty(uint256[N_COINS])
    for i in range(N_COINS):
        ideal_balance: uint256 = D1 * old_balances[i] / D0
        difference: uint256 = 0
        if ideal_balance > new_balances[i]:
            difference = ideal_balance - new_balances[i]
        else:
            difference = new_balances[i] - ideal_balance
        fees[i] = _fee * difference / FEE_DENOMINATOR
        self.balances[i] = new_balances[i] - (fees[i] * _admin_fee / FEE_DENOMINATOR)
        new_balances[i] -= fees[i]
    D2: uint256 = self.get_D_mem(new_balances, amp)

    token_amount: uint256 = (D0 - D2) * token_supply / D0
    assert token_amount != 0  # dev: zero tokens burned
    token_amount += 1  # In case of rounding errors - make it unfavorable for the "attacker"
    assert token_amount <= max_burn_amount, "Slippage screwed you"

    self.token.burnFrom(msg.sender, token_amount)  # dev: insufficient funds
    for i in range(N_COINS):
        if amounts[i] != 0:

            # "safeTransfer" which works for ERC20s which return bool or not
            _response: Bytes[32] = raw_call(
                self.coins[i],
                concat(
                    method_id("transfer(address,uint256)"),
                    convert(msg.sender, bytes32),
                    convert(amounts[i], bytes32),
                ),
                max_outsize=32,
            )  # dev: failed transfer
            if len(_response) > 0:
                assert convert(_response, bool)  # dev: failed transfer

    log RemoveLiquidityImbalance(msg.sender, amounts, fees, D1, token_supply - token_amount)


@view
@internal
def get_y_D(A_: uint256, i: int128, xp: uint256[N_COINS], D: uint256) -> uint256:
    """
    Calculate x[i] if one reduces D from being calculated for xp to D

    Done by solving quadratic equation iteratively.
    x_1**2 + x1 * (sum' - (A*n**n - 1) * D / (A * n**n)) = D ** (n + 1) / (n ** (2 * n) * prod' * A)
    x_1**2 + b*x_1 = c

    x_1 = (x_1**2 + c) / (2*x_1 + b)
    """
    # x in the input is converted to the same price/precision

    assert i >= 0  # dev: i below zero
    assert i < N_COINS  # dev: i above N_COINS

    c: uint256 = D
    S_: uint256 = 0
    Ann: uint256 = A_ * N_COINS

    _x: uint256 = 0
    for _i in range(N_COINS):
        if _i != i:
            _x = xp[_i]
        else:
            continue
        S_ += _x
        c = c * D / (_x * N_COINS)
    c = c * D / (Ann * N_COINS)
    b: uint256 = S_ + D / Ann
    y_prev: uint256 = 0
    y: uint256 = D
    for _i in range(255):
        y_prev = y
        y = (y*y + c) / (2 * y + b - D)
        # Equality with the precision of 1
        if y > y_prev:
            if y - y_prev <= 1:
                break
        else:
            if y_prev - y <= 1:
                break
    return y


@view
@internal
def _calc_withdraw_one_coin(_token_amount: uint256, i: int128) -> (uint256, uint256):
    # First, need to calculate
    # * Get current D
    # * Solve Eqn against y_i for D - _token_amount
    amp: uint256 = self._A()
    _fee: uint256 = self.fee * N_COINS / (4 * (N_COINS - 1))
    precisions: uint256[N_COINS] = PRECISION_MUL
    total_supply: uint256 = self.token.totalSupply()

    xp: uint256[N_COINS] = self._xp()

    D0: uint256 = self.get_D(xp, amp)
    D1: uint256 = D0 - _token_amount * D0 / total_supply
    xp_reduced: uint256[N_COINS] = xp

    new_y: uint256 = self.get_y_D(amp, i, xp, D1)
    dy_0: uint256 = (xp[i] - new_y) / precisions[i]  # w/o fees

    for j in range(N_COINS):
        dx_expected: uint256 = 0
        if j == i:
            dx_expected = xp[j] * D1 / D0 - new_y
        else:
            dx_expected = xp[j] - xp[j] * D1 / D0
        xp_reduced[j] -= _fee * dx_expected / FEE_DENOMINATOR

    dy: uint256 = xp_reduced[i] - self.get_y_D(amp, i, xp_reduced, D1)
    dy = (dy - 1) / precisions[i]  # Withdraw less to account for rounding errors

    return dy, dy_0 - dy


@view
@external
def calc_withdraw_one_coin(_token_amount: uint256, i: int128) -> uint256:
    return self._calc_withdraw_one_coin(_token_amount, i)[0]


@external
@nonreentrant('lock')
def remove_liquidity_one_coin(_token_amount: uint256, i: int128, min_amount: uint256):
    """
    Remove _amount of liquidity all in a form of coin i
    """
    assert not self.is_killed  # dev: is killed

    dy: uint256 = 0
    dy_fee: uint256 = 0
    dy, dy_fee = self._calc_withdraw_one_coin(_token_amount, i)
    assert dy >= min_amount, "Not enough coins removed"

    self.balances[i] -= (dy + dy_fee * self.admin_fee / FEE_DENOMINATOR)
    self.token.burnFrom(msg.sender, _token_amount)  # dev: insufficient funds

    # "safeTransfer" which works for ERC20s which return bool or not
    _response: Bytes[32] = raw_call(
        self.coins[i],
        concat(
            method_id("transfer(address,uint256)"),
            convert(msg.sender, bytes32),
            convert(dy, bytes32),
        ),
        max_outsize=32,
    )  # dev: failed transfer
    if len(_response) > 0:
        assert convert(_response, bool)  # dev: failed transfer

    log RemoveLiquidityOne(msg.sender, _token_amount, dy)


### Admin functions # allows you to change the amplification factor
@external
def ramp_A(_future_A: uint256, _future_time: uint256):
    assert msg.sender == self.owner  # dev: only owner
    assert block.timestamp >= self.initial_A_time + MIN_RAMP_TIME # make sure that we are not ramping too fast
    assert _future_time >= block.timestamp + MIN_RAMP_TIME  # dev: insufficient time

    # fetch the current A 
    _initial_A: uint256 = self._A()

    # Check that future A is in the range of [0, MAX_A]
    assert (_future_A > 0) and (_future_A < MAX_A)
    # Check that future A is not too far from initial A
    # If future A is greater than initial A, it should not be more than MAX_A_CHANGE times greater
    # If future A is less than initial A, it should not be less than MAX_A_CHANGE times less
    assert ((_future_A >= _initial_A) and (_future_A <= _initial_A * MAX_A_CHANGE)) or
           ((_future_A < _initial_A) and (_future_A * MAX_A_CHANGE >= _initial_A))
    self.initial_A = _initial_A
    self.future_A = _future_A
    self.initial_A_time = block.timestamp
    self.future_A_time = _future_time

    log RampA(_initial_A, _future_A, block.timestamp, _future_time)


# Allows to stop ramping A and sets the current A as both initial and future A. so no ramping will happen 
# @note when A -> 
@external
def stop_ramp_A():
    assert msg.sender == self.owner  # dev: only owner

    current_A: uint256 = self._A()
    self.initial_A = current_A
    self.future_A = current_A
    self.initial_A_time = block.timestamp
    self.future_A_time = block.timestamp
    # now (block.timestamp < t1) is always False, so we return saved A

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
