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


def make_card(rank: int, suit: int) -> int:
    """
    rank: 0..12 (2..A)
    suit: 0=clubs, 1=diamonds, 2=hearts, 3=spades
    """
    suit_bit = 1 << (suit + 12)
    rank_bit = 1 << (rank + 16)
    return rank_bit | suit_bit | (rank << 8) | PRIMES[rank]


def card_from_str(s: str) -> int:
    """Parse e.g. 'As', 'Th', '2c', 'Kd'."""
    s = s.strip()
    rank = RANK_CHARS.index(s[0].upper())
    suit = SUIT_CHARS.index(s[1].lower())
    return make_card(rank, suit)


def card_to_str(c: int) -> str:
    rank = (c >> 8) & 0xF
    suit_bits = (c >> 12) & 0xF
    suit = suit_bits.bit_length() - 1  # 0..3
    return RANK_CHARS[rank] + SUIT_CHARS[suit]


def card_rank(c: int) -> int:
    return (c >> 8) & 0xF


def card_suit(c: int) -> int:
    return ((c >> 12) & 0xF).bit_length() - 1


# Build full 52-card deck as a list of ints
DECK: list[int] = [make_card(r, s) for r in range(13) for s in range(4)]
