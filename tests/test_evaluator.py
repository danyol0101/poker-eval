"""
Unit tests for hand evaluator correctness.
Run: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.cards import card_from_str
from src.lookup import evaluate5, evaluate7, hand_category
from src.lookup import (
    MAX_STRAIGHT_FLUSH, MAX_FOUR_OF_A_KIND, MAX_FULL_HOUSE,
    MAX_FLUSH, MAX_STRAIGHT, MAX_THREE_OF_A_KIND, MAX_TWO_PAIR,
    MAX_ONE_PAIR, MAX_HIGH_CARD
)


def cards(*strs):
    return [card_from_str(s) for s in strs]


class TestCategories:
    def test_royal_flush(self):
        h = cards("As", "Ks", "Qs", "Js", "Ts")
        assert evaluate5(h) == 1
        assert hand_category(1) == "Royal Flush"

    def test_straight_flush(self):
        h = cards("9h", "8h", "7h", "6h", "5h")
        r = evaluate5(h)
        assert 2 <= r <= MAX_STRAIGHT_FLUSH
        assert hand_category(r) == "Straight Flush"

    def test_wheel_straight_flush(self):
        h = cards("Ah", "2h", "3h", "4h", "5h")
        r = evaluate5(h)
        assert 2 <= r <= MAX_STRAIGHT_FLUSH

    def test_four_of_a_kind(self):
        h = cards("Ah", "Ad", "Ac", "As", "Kd")
        r = evaluate5(h)
        assert MAX_STRAIGHT_FLUSH < r <= MAX_FOUR_OF_A_KIND
        assert hand_category(r) == "Four of a Kind"

    def test_full_house(self):
        h = cards("Ah", "Ad", "Ac", "Kd", "Ks")
        r = evaluate5(h)
        assert MAX_FOUR_OF_A_KIND < r <= MAX_FULL_HOUSE
        assert hand_category(r) == "Full House"

    def test_flush(self):
        h = cards("Ah", "Jh", "9h", "6h", "2h")
        r = evaluate5(h)
        assert MAX_FULL_HOUSE < r <= MAX_FLUSH
        assert hand_category(r) == "Flush"

    def test_straight(self):
        h = cards("As", "Kd", "Qh", "Jc", "Ts")
        r = evaluate5(h)
        assert MAX_FLUSH < r <= MAX_STRAIGHT
        assert hand_category(r) == "Straight"

    def test_broadway_straight(self):
        h = cards("Ah", "Kd", "Qc", "Js", "Th")
        r = evaluate5(h)
        assert r == MAX_FLUSH + 1  # Best straight (Ace-high)

    def test_wheel_straight(self):
        h = cards("Ah", "2d", "3c", "4s", "5h")
        r = evaluate5(h)
        assert MAX_FLUSH < r <= MAX_STRAIGHT

    def test_three_of_a_kind(self):
        h = cards("Ah", "Ad", "Ac", "Kd", "Qc")
        r = evaluate5(h)
        assert MAX_STRAIGHT < r <= MAX_THREE_OF_A_KIND
        assert hand_category(r) == "Three of a Kind"

    def test_two_pair(self):
        h = cards("Ah", "Ad", "Kc", "Ks", "Qd")
        r = evaluate5(h)
        assert MAX_THREE_OF_A_KIND < r <= MAX_TWO_PAIR
        assert hand_category(r) == "Two Pair"

    def test_one_pair(self):
        h = cards("Ah", "Ad", "Kc", "Qd", "Js")
        r = evaluate5(h)
        assert MAX_TWO_PAIR < r <= MAX_ONE_PAIR
        assert hand_category(r) == "One Pair"

    def test_high_card(self):
        h = cards("Ah", "Kd", "Qc", "Js", "9h")
        r = evaluate5(h)
        assert MAX_ONE_PAIR < r <= MAX_HIGH_CARD
        assert hand_category(r) == "High Card"


class TestOrdering:
    """Higher hands should have strictly lower rank values."""

    def test_royal_beats_sf(self):
        royal = evaluate5(cards("As", "Ks", "Qs", "Js", "Ts"))
        sf    = evaluate5(cards("9h", "8h", "7h", "6h", "5h"))
        assert royal < sf

    def test_sf_beats_quads(self):
        sf    = evaluate5(cards("9h", "8h", "7h", "6h", "5h"))
        quads = evaluate5(cards("Ah", "Ad", "Ac", "As", "2d"))
        assert sf < quads

    def test_quads_beats_boat(self):
        quads = evaluate5(cards("Ah", "Ad", "Ac", "As", "2d"))
        boat  = evaluate5(cards("Ah", "Ad", "Ac", "Kd", "Ks"))
        assert quads < boat

    def test_boat_beats_flush(self):
        boat  = evaluate5(cards("2h", "2d", "2c", "3d", "3s"))
        flush = evaluate5(cards("Ah", "Jh", "9h", "6h", "2h"))
        assert boat < flush

    def test_flush_beats_straight(self):
        flush    = evaluate5(cards("Ah", "Jh", "9h", "6h", "2h"))
        straight = evaluate5(cards("As", "Kd", "Qh", "Jc", "Ts"))
        assert flush < straight

    def test_aces_full_beats_kings_full(self):
        aces_full  = evaluate5(cards("Ah", "Ad", "Ac", "Kd", "Ks"))
        kings_full = evaluate5(cards("Kh", "Kd", "Kc", "Ad", "As"))
        assert aces_full < kings_full

    def test_kicker_matters_in_quads(self):
        quads_ak = evaluate5(cards("Ah", "Ad", "Ac", "As", "Kd"))
        quads_aq = evaluate5(cards("Ah", "Ad", "Ac", "As", "Qd"))
        assert quads_ak < quads_aq

    def test_two_pair_ordering(self):
        aa_kk = evaluate5(cards("Ah", "Ad", "Kc", "Ks", "Qd"))
        aa_qq = evaluate5(cards("Ah", "Ad", "Qc", "Qs", "Kd"))
        assert aa_kk < aa_qq


class TestSevenCard:
    def test_best_of_seven_finds_flush(self):
        # 7 cards containing a flush
        seven = cards("Ah", "Kh", "Qh", "Jh", "Th", "2d", "3s")
        r = evaluate7(seven)
        assert r == 1  # Royal flush

    def test_hidden_straight_flush(self):
        seven = cards("9h", "8h", "7h", "6h", "5h", "Ah", "2d")
        r = evaluate7(seven)
        assert hand_category(r) == "Straight Flush"

    def test_best_pair_chosen(self):
        # Board pairs the ace, player has AA — should find full house or quads
        seven = cards("Ah", "Ac", "Ad", "As", "Kd", "2h", "3s")
        r = evaluate7(seven)
        assert hand_category(r) == "Four of a Kind"


class TestEquity:
    def test_equity_sums_to_one(self):
        from src.equity import equity
        from src.cards import card_from_str
        h1 = [card_from_str("As"), card_from_str("Kd")]
        h2 = [card_from_str("Qh"), card_from_str("Jh")]
        result = equity([h1, h2], n_trials=10_000, seed=42)
        total = sum(result['equity'])
        assert abs(total - 1.0) < 1e-9

    def test_aa_beats_22_preflop(self):
        from src.equity import equity
        from src.cards import card_from_str
        aa = [card_from_str("As"), card_from_str("Ah")]
        two = [card_from_str("2d"), card_from_str("2c")]
        result = equity([aa, two], n_trials=50_000, seed=42)
        # AA is ~83% vs 22
        assert result['equity'][0] > 0.75

    def test_exact_equity_river(self):
        from src.equity import equity_exact
        from src.cards import card_from_str
        # River: 1 card left, deterministic
        h1 = [card_from_str("As"), card_from_str("Kd")]
        h2 = [card_from_str("Qh"), card_from_str("Jh")]
        board = [card_from_str(c) for c in ["Ah", "2d", "3c", "4s"]]
        result = equity_exact([h1, h2], board)
        assert result['exact']
        total = sum(result['equity'])
        assert abs(total - 1.0) < 1e-9

    def test_dominated_hand(self):
        from src.equity import equity_exact
        from src.cards import card_from_str
        # On the river when one hand is already beaten
        h1 = [card_from_str("As"), card_from_str("Ad")]  # pair of aces
        h2 = [card_from_str("2h"), card_from_str("2d")]  # pair of 2s
        board = [card_from_str(c) for c in ["Ah", "Ac", "Kd", "Qs", "Jh"]]  # full board
        result = equity_exact([h1, h2], board)
        # h1 has AAAA + K vs h2 has 2s pair — h1 wins all
        assert result['equity'][0] == 1.0
        assert result['equity'][1] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
