"""
Card representation.

Each card is a 32-bit integer with the following layout (Cactus Kev encoding):

  Bits 31-16: rank bitmask (bit 16+rank set, for flush detection)
  Bits 15-12: suit (one-hot: 1000=spades, 0100=hearts, 0010=diamonds, 0001=clubs)
  Bits 11- 8: rank 0-12 (2=0, 3=1, ..., A=12)
  Bits  7- 0: prime number for this rank (used in product hashing)

Primes by rank: 2,3,5,7,11,13,17,19,23,29,31,37,41
"""

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"

# Primes for ranks 2..A — product uniquely identifies any 5-card rank combination
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]

SUIT_BITS = {
    'c': 0x1000,  # clubs    bit 12
    'd': 0x2000,  # diamonds bit 13
    'h': 0x4000,  # hearts   bit 14
    's': 0x8000,  # spades   bit 15
}
