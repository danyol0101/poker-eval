"""
Hand rank lookup tables.

Strategy
--------
A 5-card hand is evaluated to a value in [1, 7462] where:
  1     = Royal Flush (best)
  7462  = 7-2 offsuit (worst)

Two separate lookup paths:
  1. FLUSH table:  key = rank bitmask (13 bits)   → hand rank
  2. UNIQUE5 table: key = prime product            → hand rank  (straights + high cards)
  3. Non-unique non-flush: key = prime product mod big prime → hand rank

Hand category boundaries (lower = better):
  Royal Flush:      1
  Straight Flushes: 2 – 10
  Four of a Kind:   11 – 166
  Full House:       167 – 322
  Flush:            323 – 1599
  Straight:         1600 – 1609
  Three of a Kind:  1610 – 2467
  Two Pair:         2468 – 3325
  One Pair:         3326 – 6185
  High Card:        6186 – 7462
"""

from itertools import combinations
from .cards import PRIMES, RANK_CHARS, make_card

# ---------------------------------------------------------------------------
# Hand category thresholds
# ---------------------------------------------------------------------------
MAX_STRAIGHT_FLUSH = 10
MAX_FOUR_OF_A_KIND  = 166
MAX_FULL_HOUSE      = 322
MAX_FLUSH           = 1599
MAX_STRAIGHT        = 1609
MAX_THREE_OF_A_KIND = 2467
MAX_TWO_PAIR        = 3325
MAX_ONE_PAIR        = 6185
MAX_HIGH_CARD       = 7462


def hand_category(rank: int) -> str:
    if rank == 1:                         return "Royal Flush"
    if rank <= MAX_STRAIGHT_FLUSH:        return "Straight Flush"
    if rank <= MAX_FOUR_OF_A_KIND:        return "Four of a Kind"
    if rank <= MAX_FULL_HOUSE:            return "Full House"
    if rank <= MAX_FLUSH:                 return "Flush"
    if rank <= MAX_STRAIGHT:             return "Straight"
    if rank <= MAX_THREE_OF_A_KIND:       return "Three of a Kind"
    if rank <= MAX_TWO_PAIR:              return "Two Pair"
    if rank <= MAX_ONE_PAIR:              return "One Pair"
    return "High Card"


# ---------------------------------------------------------------------------
# Generate all 5-card hand ranks
# ---------------------------------------------------------------------------

def _build_tables():
    """
    Returns (flush_table, unique5_table, multiples_table).

    flush_table:    rank_bitmask (int) -> hand_rank
    unique5_table:  prime_product (int) -> hand_rank   (straights + unpaired high cards)
    multiples_table: prime_product (int) -> hand_rank  (pairs, trips, quads, boats)
    """
    flush_table: dict[int, int] = {}
    unique5_table: dict[int, int] = {}
    multiples_table: dict[int, int] = {}

    rank_counter = 0

    def next_rank():
        nonlocal rank_counter
        rank_counter += 1
        return rank_counter

    # --- Straight Flushes (10 total including Royal) ---
    # Ranks: A-high (ranks 12,11,10,9,8) down to A-low (12,0,1,2,3)
    sf_ranks = [
        (12, 11, 10, 9, 8),  # Royal
        (11, 10,  9, 8, 7),
        (10,  9,  8, 7, 6),
        ( 9,  8,  7, 6, 5),
        ( 8,  7,  6, 5, 4),
        ( 7,  6,  5, 4, 3),
        ( 6,  5,  4, 3, 2),
        ( 5,  4,  3, 2, 1),
        ( 4,  3,  2, 1, 0),
        (12,  3,  2, 1, 0),  # A-2-3-4-5 (wheel)
    ]
    # Track straight bitmasks separately for high-card exclusion
    straight_bitmasks: set[int] = set()
    for hand in sf_ranks:
        bitmask = sum(1 << r for r in hand)
        straight_bitmasks.add(bitmask)
        hr = next_rank()
        flush_table[bitmask] = hr

    # --- Four of a Kind (156 = 13 quads × 12 kickers) ---
    for quad_rank in range(12, -1, -1):
        for kicker in range(12, -1, -1):
            if kicker == quad_rank:
                continue
            prod = PRIMES[quad_rank]**4 * PRIMES[kicker]
            multiples_table[prod] = next_rank()

    # --- Full House (156 = 13 trips × 12 pairs) ---
    for trip_rank in range(12, -1, -1):
        for pair_rank in range(12, -1, -1):
            if pair_rank == trip_rank:
                continue
            prod = PRIMES[trip_rank]**3 * PRIMES[pair_rank]**2
            multiples_table[prod] = next_rank()

    # --- Flushes (1277 non-straight flush suited combos) ---
    # All C(13,5) = 1287 bitmasks, minus 10 straight-flush bitmasks
    for combo in combinations(range(13), 5):
        bitmask = sum(1 << r for r in combo)
        if bitmask in flush_table:
            continue  # already a straight flush
        # Not a straight: check no 5 consecutive
        flush_table[bitmask] = next_rank()

    # --- Straights (10, same structure as SF but not flush) ---
    for hand in sf_ranks:
        prod = 1
        for r in hand:
            prod *= PRIMES[r]
        unique5_table[prod] = next_rank()

    # --- Three of a Kind (858) ---
    for trip_rank in range(12, -1, -1):
        remaining = [r for r in range(13) if r != trip_rank]
        for k1, k2 in combinations(remaining, 2):
            prod = PRIMES[trip_rank]**3 * PRIMES[k1] * PRIMES[k2]
            multiples_table[prod] = next_rank()

    # --- Two Pair (858) ---
    for p1 in range(12, -1, -1):
        for p2 in range(p1 - 1, -1, -1):
            for kicker in range(12, -1, -1):
                if kicker == p1 or kicker == p2:
                    continue
                prod = PRIMES[p1]**2 * PRIMES[p2]**2 * PRIMES[kicker]
                multiples_table[prod] = next_rank()

    # --- One Pair (2860) ---
    for pair_rank in range(12, -1, -1):
        remaining = [r for r in range(13) if r != pair_rank]
        for k1, k2, k3 in combinations(remaining, 3):
            prod = PRIMES[pair_rank]**2 * PRIMES[k1] * PRIMES[k2] * PRIMES[k3]
            multiples_table[prod] = next_rank()

    # --- High Card (1277 — same rank combos as flushes, different lookup path) ---
    for combo in combinations(range(13), 5):
        bitmask = sum(1 << r for r in combo)
        if bitmask in straight_bitmasks:
            continue  # already a straight — in unique5_table
        prod = 1
        for r in combo:
            prod *= PRIMES[r]
        unique5_table[prod] = next_rank()

    return flush_table, unique5_table, multiples_table


# Build once at import time
FLUSH_TABLE, UNIQUE5_TABLE, MULTIPLES_TABLE = _build_tables()


def evaluate5(cards: list[int]) -> int:
    """
    Evaluate a 5-card hand. Returns rank in [1, 7462] (lower = better).
    cards: list of 5 card ints from cards.py
    """
    # Check flush: all 4 suit bits must match
    suit_intersection = cards[0] & cards[1] & cards[2] & cards[3] & cards[4] & 0xF000
    if suit_intersection:
        # Flush path: use rank bitmask
        bitmask = 0
        for c in cards:
            bitmask |= (c >> 16)
        return FLUSH_TABLE[bitmask]

    # Non-flush: compute prime product
    prod = 1
    for c in cards:
        prod *= c & 0xFF  # low byte = prime

    if prod in UNIQUE5_TABLE:
        return UNIQUE5_TABLE[prod]
    return MULTIPLES_TABLE[prod]
