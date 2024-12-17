"""
Hand rank lookup tables.

A 5-card hand is evaluated to a value in [1, 7462] where:
  1     = Royal Flush (best)
  7462  = 7-2 offsuit (worst)

Two separate lookup paths:
  1. FLUSH table:  key = rank bitmask (13 bits)   → hand rank
  2. UNIQUE5 table: key = prime product            → hand rank  (straights + high cards)
  3. Non-unique non-flush: key = prime product     → hand rank

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
