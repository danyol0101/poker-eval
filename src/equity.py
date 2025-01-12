"""
Monte Carlo equity engine.

Given hole cards for N players and optional board cards (0..5 community),
estimate each player's win/tie probability by sampling random runouts.

Speed tricks:
  - Pre-remove known cards from deck once, not per simulation
  - Use numpy for fast shuffling of the remaining deck
  - Batch evaluate: fill hands array in-place per trial
  - Pure Python evaluate7 is ~2µs/call; at 10k trials/sec this is ~100ms for 100k sims
"""

import random
import time
from typing import Optional
import numpy as np

from .cards import DECK, card_from_str, card_to_str
from .lookup import evaluate7, hand_category


def _remove_known(cards_to_remove: list[int]) -> list[int]:
    """Return deck with specified cards removed."""
    remove_set = set(cards_to_remove)
    return [c for c in DECK if c not in remove_set]


def equity(
    hole_cards: list[list[int]],   # [[c1,c2], [c1,c2], ...]  one pair per player
    board: list[int] = None,       # 0–5 community cards already dealt
    n_trials: int = 100_000,
    seed: Optional[int] = None,
) -> dict:
    """
    Monte Carlo equity calculation.

    Returns dict with keys:
      'wins':  list of win counts per player
      'ties':  list of tie counts per player (split pots)
      'equity': list of equity fractions (wins + ties/n_players) / n_trials
      'trials': actual number of trials run
      'elapsed_ms': wall time
    """
    if board is None:
        board = []

    n_players = len(hole_cards)
    n_board = len(board)
    n_to_deal = 5 - n_board  # cards still needed on board

    # Build dead-card-free deck
    known = [c for pair in hole_cards for c in pair] + board
    live_deck = np.array(_remove_known(known), dtype=np.int64)
    n_live = len(live_deck)

    rng = np.random.default_rng(seed)

    wins = [0] * n_players
    ties = [0] * n_players

    t0 = time.perf_counter()

    for _ in range(n_trials):
        # Fast shuffle: draw n_to_deal cards without replacement
        idx = rng.choice(n_live, size=n_to_deal, replace=False)
        runout = live_deck[idx].tolist()

        full_board = board + runout

        # Evaluate each player's best 7-card hand
        ranks = []
        for pair in hole_cards:
            seven = pair + full_board
            ranks.append(evaluate7(seven))

        best = min(ranks)
        winners = [i for i, r in enumerate(ranks) if r == best]

        if len(winners) == 1:
            wins[winners[0]] += 1
        else:
            for w in winners:
                ties[w] += 1

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Equity: credit ties as fractional wins
    equity_vals = [
        (wins[i] + ties[i] / n_players) / n_trials
        for i in range(n_players)
    ]

    return {
        'wins':       wins,
        'ties':       ties,
        'equity':     equity_vals,
        'trials':     n_trials,
        'elapsed_ms': elapsed_ms,
    }


def parse_hand(s: str) -> list[int]:
    """Parse '  AsKh ' -> [card_int, card_int]."""
    tokens = s.strip().split()
    return [card_from_str(t) for t in tokens]


def parse_board(s: str) -> list[int]:
    """Parse 'Ah 2d 3c' -> list of card ints."""
    if not s.strip():
        return []
    return [card_from_str(t) for t in s.strip().split()]
