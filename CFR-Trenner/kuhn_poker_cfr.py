"""
Kuhn Poker CFR (Counterfactual Regret Minimization)
(2-player version)

adapted from [Trenner2020, Part 6] 
https://medium.com/ai-in-plain-english/building-a-poker-ai-part-6-beating-kuhn-poker-with-cfr-using-python-1b4172a6ab2d 

Change WK: simplified code
    - by using only a scalar reach probability per node, not a reach probability vector (player 0,1)
    - by using defaultdict for KuhnCFRTrainer.infoset_map (simplifies get_information_set)
"""
from typing import List, Dict
from collections import defaultdict
import random
import numpy as np
import sys
import matplotlib.pyplot as plt

Actions = ['B', 'C']  # B: bet in bet-check-situation; call in a call-fold-situation (the more aggressive action)
                      # C: check in bet-check-situation; fold in a call-fold-situation (the more defensive action)

class InformationSet():
    def __init__(self):
        self.cumulative_regrets = np.zeros(shape=len(Actions))+1e-8
        self.strategy_sum = np.zeros(shape=len(Actions))+1e-8
        self.num_actions = len(Actions)

    def normalize(self, strategy: np.array) -> np.array:
        """Normalize a strategy. If there are no positive regrets, use a uniform random strategy"""
        return strategy/sum(strategy)
        #if sum(strategy) > 0:
        #    strategy /= sum(strategy)
        #else:
        #    strategy = np.array([1.0 / self.num_actions] * self.num_actions)
        #return strategy

    def get_strategy(self, reach_probability: float) -> np.array:
        """Return regret-matching strategy"""
        strategy = np.maximum(0, self.cumulative_regrets)
        strategy = self.normalize(strategy)

        self.strategy_sum += reach_probability * strategy
        return strategy

    def get_average_strategy(self) -> np.array:
        return self.normalize(self.strategy_sum.copy())


class KuhnPoker():
    @staticmethod
    def is_terminal(history: str) -> bool:
        return history in ['BC', 'BB', 'CC', 'CBB', 'CBC']

    @staticmethod
    def get_payoff(history: str, cards: List[str]) -> int:
        """get payoff for 'active' player in terminal history"""
        if history in ['BC', 'CBC']:
            return +1
        else:  # CC or BB or CBB
            payoff = 2 if 'B' in history else 1
            active_player = len(history) % 2
            player_card = cards[active_player]
            opponent_card = cards[(active_player + 1) % 2]
            if player_card == 'K' or opponent_card == 'J':
                return payoff
            else:
                return -payoff


class KuhnCFRTrainer():
    def __init__(self):
        self.infoset_map = defaultdict(InformationSet)

    def get_information_set(self, card_and_history: str) -> InformationSet:
        """add if needed and return"""
        return self.infoset_map[card_and_history]

    def cfr(self, cards: List[str], history: str, reach_probability: float, active_player: int):
        if KuhnPoker.is_terminal(history):
            return KuhnPoker.get_payoff(history, cards)

        my_card = cards[active_player]
        info_set = self.get_information_set(my_card + history)

        strategy = info_set.get_strategy(reach_probability)
        opponent = (active_player + 1) % 2
        counterfactual_values = np.zeros(len(Actions))

        for ix, action in enumerate(Actions):
            action_probability = strategy[ix]

            # compute new reach probability after this action
            new_reach_probability = reach_probability * action_probability

            # recursively call cfr method, next player to act is the opponent
            counterfactual_values[ix] = -self.cfr(cards, history + action, new_reach_probability, opponent)

        # Value of the current game state is just counterfactual values weighted by action probabilities
        node_value = counterfactual_values.dot(strategy)
        for ix, action in enumerate(Actions):
            info_set.cumulative_regrets[ix] += reach_probability * (counterfactual_values[ix] - node_value)

        return node_value

    def train(self, num_iterations: int) -> int:
        util = 0
        kuhn_cards = ['J', 'Q', 'K']
        for k in range(num_iterations):
            cards = random.sample(kuhn_cards, 2)
            history = ''
            reach_probability = 1 
            util += self.cfr(cards, history, reach_probability, 0)
            if k%skip==0:
                j_strategy = self.get_information_set('J').get_average_strategy()
                j_k_bet_ratio = j_strategy[0] / self.get_information_set('K').get_average_strategy()[0]
                strat_arr[k//skip] = np.hstack((j_strategy, j_k_bet_ratio, 1/3.))
                # 4 curves: P('JB') blue, 1-P('JB') orange, P('JB')/P('KB') green, 1/3 red (the ideal green)
        return util


if __name__ == "__main__":
    if len(sys.argv) < 2:
        num_iterations = 50000
    else:
        num_iterations = int(sys.argv[1])
    np.set_printoptions(precision=3, floatmode='fixed', suppress=True)
    random.seed(44)

    skip = 100
    # strat_arr is only used for plotting the evolution of the strategy:
    strat_arr = np.zeros(shape=(int(num_iterations/skip),len(Actions)+2))
    cfr_trainer = KuhnCFRTrainer()
    util = cfr_trainer.train(num_iterations)

    print(f"\nRunning Kuhn Poker chance sampling CFR for {num_iterations} iterations")
    print(f"\nExpected average game value (for player 1): {(-1./18):.3f}")
    print(f"Computed average game value               : {(util / num_iterations):.3f}\n")

    print("We expect the bet frequency for a Jack (blue curve) to be between 0 and 1/3")
    print("The bet frequency of a King should be three times the one for a Jack (green curve approaches red line)\n")

    print(f"History  Bet   Pass")
    for name, info_set in sorted(cfr_trainer.infoset_map.items(), key=lambda s: len(s[0])):
        print(f"{name:3}:    {info_set.get_average_strategy()}")

    plt.plot(strat_arr)
    plt.show()
