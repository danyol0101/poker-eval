"""
Numba-accelerated 7-card evaluator and Monte Carlo engine.

Key speed techniques:
  1. Flat numpy array for flush lookup (8192-entry direct index by 13-bit bitmask)
  2. Sorted arrays + binary search for prime-product lookup (O(log 6175) = 13 steps)
  3. @njit on the entire hot path — evaluate5, evaluate7, MC loop
  4. Fisher-Yates partial shuffle inside numba for runout sampling
"""

import numpy as np
from numba import njit
import time
from itertools import combinations as _comb

from .lookup import FLUSH_TABLE, UNIQUE5_TABLE, MULTIPLES_TABLE
from .cards import DECK


# ---------------------------------------------------------------------------
# Build flat lookup arrays
# ---------------------------------------------------------------------------

_HASH_MAGIC = np.int64(-0x61C8864680B583EB)  # Fibonacci hash constant (odd, 64-bit)
_HASH_BITS  = 14                              # 2^14 = 16384 slots, load ≈ 0.38
_HASH_SIZE  = 1 << _HASH_BITS                # must be power-of-2


def _build_tables():
    # FLUSH TABLE: direct index by 13-bit bitmask (max key = 8191)
    flush_arr = np.zeros(8192, dtype=np.int32)
    for bitmask, rank in FLUSH_TABLE.items():
        flush_arr[bitmask] = rank

    # Combined unique5 + multiples → open-addressing hash table
    all_products = {}
    all_products.update(UNIQUE5_TABLE)
    all_products.update(MULTIPLES_TABLE)

    hash_keys = np.zeros(_HASH_SIZE, dtype=np.int64)   # 0 = empty sentinel
    hash_vals = np.zeros(_HASH_SIZE, dtype=np.int32)
    mask = np.int64(_HASH_SIZE - 1)

    import ctypes
    for prod, rank in all_products.items():
        # 64-bit wraparound multiply (same behaviour as numba/C)
        p64 = ctypes.c_int64(prod).value
        m64 = ctypes.c_int64(int(_HASH_MAGIC)).value
        slot = int(ctypes.c_int64(p64 * m64).value >> (64 - _HASH_BITS)) & int(mask)
        while hash_keys[slot] != 0:
            slot = (slot + 1) & int(mask)
        hash_keys[slot] = p64
        hash_vals[slot] = rank

    return flush_arr, hash_keys, hash_vals


print("Building fast lookup tables...", end=" ", flush=True)
_t0 = time.perf_counter()
FLUSH_ARR, HASH_KEYS, HASH_VALS = _build_tables()
print(f"done ({(time.perf_counter()-_t0)*1000:.0f}ms, hash_size={_HASH_SIZE})")

# C(7,5) = 21 combinations as numpy array for numba
_COMBOS = np.array(list(_comb(range(7), 5)), dtype=np.int32)  # (21, 5)


# ---------------------------------------------------------------------------
# JIT-compiled evaluator
# ---------------------------------------------------------------------------

_HASH_MAGIC_NB = np.int64(-0x61C8864680B583EB)
_HASH_MASK_NB  = np.int64(_HASH_SIZE - 1)
_HASH_SHIFT_NB = np.int64(64 - _HASH_BITS)


@njit(cache=True)
def _hash_lookup(prod, hash_keys, hash_vals):
    """O(1) open-addressing hash table lookup. prod is guaranteed to exist."""
    slot = int((np.int64(prod) * _HASH_MAGIC_NB) >> _HASH_SHIFT_NB) & int(_HASH_MASK_NB)
    while hash_keys[slot] != prod:
        slot = (slot + 1) & int(_HASH_MASK_NB)
    return hash_vals[slot]


@njit(cache=True)
def _eval5(c0, c1, c2, c3, c4, flush_arr, hash_keys, hash_vals):
    """Evaluate 5 cards → rank in [1, 7462]. Lower = better."""
    # Flush: all 5 share a suit bit
    suit_bits = c0 & c1 & c2 & c3 & c4 & 0xF000
    if suit_bits:
        bitmask = ((c0 | c1 | c2 | c3 | c4) >> 16) & 0x1FFF
        return flush_arr[bitmask]

    # Non-flush: hash table lookup on prime product
    prod = (c0 & 0xFF) * (c1 & 0xFF) * (c2 & 0xFF) * (c3 & 0xFF) * (c4 & 0xFF)
    return _hash_lookup(prod, hash_keys, hash_vals)


@njit(cache=True)
def _eval7(cards, flush_arr, hash_keys, hash_vals, combos):
    """Best 5-card rank from 7 cards."""
    best = 9999
    for i in range(21):
        r = _eval5(
            cards[combos[i, 0]], cards[combos[i, 1]], cards[combos[i, 2]],
            cards[combos[i, 3]], cards[combos[i, 4]],
            flush_arr, hash_keys, hash_vals
        )
        if r < best:
            best = r
    return best


@njit(cache=True)
def _mc_2player(
    hole0, hole1, board, live_deck, n_trials,
    flush_arr, hash_keys, hash_vals, combos, seed,
):
    """Hot path: heads-up Monte Carlo. Returns (wins0, wins1, ties)."""
    np.random.seed(seed)
    wins0 = wins1 = ties = 0

    n_live  = len(live_deck)
    n_board = len(board)
    n_deal  = 5 - n_board

    deck  = live_deck.copy()
    seven = np.empty(7, dtype=np.int64)

    for _ in range(n_trials):
        # Partial Fisher-Yates: swap n_deal elements to front
        for k in range(n_deal):
            j = k + (np.random.randint(0, n_live - k))
            deck[k], deck[j] = deck[j], deck[k]

        # Player 0
        seven[0] = hole0[0]; seven[1] = hole0[1]
        for k in range(n_board):    seven[2 + k]         = board[k]
        for k in range(n_deal):     seven[2 + n_board + k] = deck[k]
        r0 = _eval7(seven, flush_arr, hash_keys, hash_vals, combos)

        # Player 1
        seven[0] = hole1[0]; seven[1] = hole1[1]
        r1 = _eval7(seven, flush_arr, hash_keys, hash_vals, combos)

        if   r0 < r1: wins0 += 1
        elif r1 < r0: wins1 += 1
        else:         ties  += 1

    return wins0, wins1, ties


@njit(cache=True)
def _mc_nplayer(
    hole_cards, board, live_deck, n_trials,
    flush_arr, hash_keys, hash_vals, combos, seed,
):
    """General N-player Monte Carlo."""
    np.random.seed(seed)
    n_players = hole_cards.shape[0]
    wins = np.zeros(n_players, dtype=np.int64)
    ties = np.zeros(n_players, dtype=np.int64)

    n_live  = len(live_deck)
    n_board = len(board)
    n_deal  = 5 - n_board

    deck  = live_deck.copy()
    seven = np.empty(7, dtype=np.int64)
    ranks = np.empty(n_players, dtype=np.int32)

    for _ in range(n_trials):
        for k in range(n_deal):
            j = k + (np.random.randint(0, n_live - k))
            deck[k], deck[j] = deck[j], deck[k]

        for p in range(n_players):
            seven[0] = hole_cards[p, 0]; seven[1] = hole_cards[p, 1]
            for k in range(n_board):    seven[2 + k]          = board[k]
            for k in range(n_deal):     seven[2 + n_board + k] = deck[k]
            ranks[p] = _eval7(seven, flush_arr, hash_keys, hash_vals, combos)

        best = 9999
        for p in range(n_players):
            if ranks[p] < best: best = ranks[p]

        n_win = 0
        for p in range(n_players):
            if ranks[p] == best: n_win += 1

        for p in range(n_players):
            if ranks[p] == best:
                if n_win == 1: wins[p] += 1
                else:          ties[p] += 1

    return wins, ties


# ---------------------------------------------------------------------------
# Warm up JIT on first import
# ---------------------------------------------------------------------------

def _warmup():
    deck_np = np.array(DECK, dtype=np.int64)
    h0   = deck_np[:2].copy()
    h1   = deck_np[2:4].copy()
    brd  = np.array([], dtype=np.int64)
    live = deck_np[4:].copy()
    _mc_2player(h0, h1, brd, live, 1, FLUSH_ARR, HASH_KEYS, HASH_VALS, _COMBOS, 0)


print("Warming up JIT...", end=" ", flush=True)
_t1 = time.perf_counter()
_warmup()
print(f"done ({(time.perf_counter()-_t1)*1000:.0f}ms)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def equity_fast(
    hole_cards: list[list[int]],
    board: list[int] = None,
    n_trials: int = 100_000,
    seed: int = 42,
) -> dict:
    """Fast equity calculation via numba JIT Monte Carlo."""
    if board is None:
        board = []

    n_players = len(hole_cards)
    known     = set(c for pair in hole_cards for c in pair) | set(board)
    live_deck = np.array([c for c in DECK if c not in known], dtype=np.int64)
    board_arr = np.array(board, dtype=np.int64)

    t0 = time.perf_counter()

    if n_players == 2:
        h0 = np.array(hole_cards[0], dtype=np.int64)
        h1 = np.array(hole_cards[1], dtype=np.int64)
        w0, w1, t = _mc_2player(
            h0, h1, board_arr, live_deck, n_trials,
            FLUSH_ARR, HASH_KEYS, HASH_VALS, _COMBOS, seed,
        )
        wins = [int(w0), int(w1)]
        tie_c = [int(t), int(t)]
    else:
        hc = np.array(hole_cards, dtype=np.int64)
        w_arr, t_arr = _mc_nplayer(
            hc, board_arr, live_deck, n_trials,
            FLUSH_ARR, HASH_KEYS, HASH_VALS, _COMBOS, seed,
        )
        wins = [int(x) for x in w_arr]
        tie_c = [int(x) for x in t_arr]

    elapsed_ms = (time.perf_counter() - t0) * 1000
    equity_vals = [(wins[i] + tie_c[i] / n_players) / n_trials for i in range(n_players)]

    return {
        'wins':       wins,
        'ties':       tie_c,
        'equity':     equity_vals,
        'trials':     n_trials,
        'elapsed_ms': elapsed_ms,
    }
