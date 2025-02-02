#!/usr/bin/env python3
"""
Poker equity calculator CLI.

Usage examples:

  # Heads-up preflop
  python cli.py "As Kd" "Qh Jh"

  # With a flop
  python cli.py "As Kd" "2h 7c" --board "Ah 2d 3c"

  # Three players, turn dealt, exact enumeration
  python cli.py "As Kd" "Qh Jh" "Tc 9c" --board "Ah 2d 3c 4s" --exact

  # Control simulation count
  python cli.py "As Kd" "Qh Jh" --trials 500000
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src import (
    card_from_str, card_to_str, evaluate7, hand_category,
    equity, equity_exact, parse_hand, parse_board
)
from src.fast_eval import equity_fast
from src.lookup import MAX_STRAIGHT_FLUSH, MAX_FOUR_OF_A_KIND, MAX_FULL_HOUSE


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def format_equity(e: float) -> str:
    pct = e * 100
    if pct >= 60:
        c = "92"   # green
    elif pct >= 40:
        c = "93"   # yellow
    else:
        c = "91"   # red
    return color(f"{pct:6.2f}%", c)


def main():
    parser = argparse.ArgumentParser(
        description="Poker hand equity calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("hands", nargs="+", help="Hole cards per player, e.g. 'As Kd'")
    parser.add_argument("--board", default="", help="Community cards, e.g. 'Ah 2d 3c'")
    parser.add_argument("--trials", type=int, default=100_000, help="Monte Carlo trials (default 100k)")
    parser.add_argument("--exact", action="store_true", help="Exact enumeration (only for few runouts)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    args = parser.parse_args()

    # Parse inputs
    try:
        hole_cards = [parse_hand(h) for h in args.hands]
        board = parse_board(args.board)
    except (ValueError, IndexError) as e:
        print(f"Error parsing cards: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate no duplicate cards
    all_cards = [c for pair in hole_cards for c in pair] + board
    seen = set()
    for c in all_cards:
        from src.cards import card_to_str as _c2s
        s = _c2s(c)
        if s in seen:
            print(f"Error: duplicate card {s}", file=sys.stderr)
            sys.exit(1)
        seen.add(s)

    n_players = len(hole_cards)
    n_board = len(board)

    if n_board > 5:
        print("Board can have at most 5 cards.", file=sys.stderr)
        sys.exit(1)

    # Display inputs
    print()
    print(color("  POKER EQUITY CALCULATOR", "1;36"))
    print(color("  " + "─" * 40, "36"))
    print()

    labels = [f"Player {i+1}" for i in range(n_players)]
    for i, (label, cards) in enumerate(zip(labels, hole_cards)):
        hand_str = " ".join(card_to_str(c) for c in cards)
        if n_board == 5:
            r = evaluate7(cards + board)
            cat = hand_category(r)
            print(f"  {label}: {color(hand_str, '1;37')}  →  {color(cat, '33')}  (rank {r})")
        else:
            print(f"  {label}: {color(hand_str, '1;37')}")

    if board:
        board_str = " ".join(card_to_str(c) for c in board)
        stage = {0: "Preflop", 3: "Flop", 4: "Turn", 5: "River"}[n_board]
        print(f"\n  Board ({stage}): {color(board_str, '1;34')}")
    else:
        print(f"\n  Board: (preflop)")

    print()

    # Run equity calculation
    n_to_deal = 5 - n_board
    total_runouts = 1
    live_remaining = 52 - 2 * n_players - n_board
    for i in range(n_to_deal):
        total_runouts *= (live_remaining - i)
    for i in range(1, n_to_deal + 1):
        total_runouts //= i

    method = "exact" if args.exact else "Monte Carlo"
    print(f"  Runouts: {total_runouts:,}  |  Method: {method}", end="")
    if not args.exact:
        print(f"  |  Trials: {args.trials:,}", end="")
    print("\n")

    try:
        if args.exact:
            result = equity_exact(hole_cards, board)
        else:
            seed = args.seed if args.seed is not None else 42
            result = equity_fast(hole_cards, board, n_trials=args.trials, seed=seed)
    except ValueError as e:
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    # Display results
    print(color("  RESULTS", "1;36"))
    print(color("  " + "─" * 40, "36"))
    print(f"  {'Player':<12} {'Equity':>8}  {'Wins':>8}  {'Ties':>8}")
    print(f"  {'──────':<12} {'──────':>8}  {'────':>8}  {'────':>8}")

    for i, label in enumerate(labels):
        eq  = format_equity(result['equity'][i])
        win = f"{result['wins'][i]:,}"
        tie = f"{result['ties'][i]:,}"
        print(f"  {label:<12} {eq}  {win:>8}  {tie:>8}")

    n_trials = result['trials']
    elapsed  = result['elapsed_ms']
    speed    = n_trials / (elapsed / 1000)
    print()
    print(f"  {n_trials:,} trials in {elapsed:.1f}ms  ({speed:,.0f} evals/sec)")
    print()


if __name__ == "__main__":
    main()
