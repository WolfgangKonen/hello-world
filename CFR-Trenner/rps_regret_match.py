"""
    Demonstration of Regret Matching for the game RPS (Rock-Paper-Scissors).
    
    Different strategies can be selected:
    1)  To learn the best strategy against a fixed opponent, use CASE="FIXED".
        and enter values for fixed_opp_strategy. If the opponent has one probability 
        larger than the two others, the best strategy is then to *always* select 
        that action that beats the most probable opponent action.
    2)  To learn the Nash equilibrium strategy by self-play, use CASE="SELFPLAY": 
        The three action probabilites approach (1/3,1/3,1/3) 
    3)  To let both agents do **independent** regret matching, use CASE="INDEPEND":
        This gives the same result as CASE="SELFPLAY", but it simulates the case
        that opponent does regret matching not knowing what our strategy is.  
    4)  In CASE=="STEAL" the opponent tries to be clever: She assumes that we 
        do regret matching, 'steals' our strategy calculation and perfoms always
        that action that wins against our most probable action. Again the action 
        probabilities approach the Nash equilibrium (1/3,1/3,1/3), however 
        much smoother than in case 2) or 3). 
        The Nash equilibrium emerges because our agent notices the opponent 
        strategy and adapts to it.
        Action(0) is always slightly lower (0.2-1.5%) in probability than the 
        two others, unclear why. Additionally, our agent has a slight, but
        always negative payoff.
        
    If in case "FIXED" the opponent has a fixed strategy close to or equal to
    the Nash equilibrium (1/3,1/3,1/3), the learnt strategy is erratic (different 
    runs lead to different probabilities). But this is OK, since our agent has 
    always a low average payoff and it is simply because **each** strategy achieves 
    the same result against a (1/3,1/3,1/3)-opponent.
    
    This code is assembled from [Trenner2019, Part 4] 
    https://ai.plainenglish.io/steps-to-building-a-poker-ai-part-4-regret-matching-for-rock-paper-scissors-in-python-168411edbb13 
    with a few extensions by WK
        - different cases CASE, as explained above
        - fixing the non-negative regrets issue in get_regrets
"""
from enum import Enum
from typing import List
import random
import numpy as np
import matplotlib.pyplot as plt

CASE = "SELFPLAY"   # "FIXED"  "SELFPLAY"  "INDEPEND"   "STEAL"
CUMULATE_STRAT = False              # can be False after fixing the non-negative regrets issue
num_iterations = 20000


class Action(Enum):
  ROCK = 0
  PAPER = 1
  SCISSORS = 2
  # PAPER beats ROCK, SCISSOR beats PAPER and ROCK beats SCISSOR


def get_payoff(action_1: Action, action_2: Action) -> int:
    """
    :param action_1: action of player 1
    :param action_2: action of player 2
    :return: the payoff for player 1
    """
    payoff = np.array([0, 1, -1])
    mod3_val = (action_1.value - action_2.value) % 3
    return payoff[mod3_val]


def get_strategy(cumulative_regrets: np.array) -> np.array:
    """Return regret-matching strategy"""
    pos_cumulative_regrets = np.maximum(1e-8, cumulative_regrets)
    return pos_cumulative_regrets / sum(pos_cumulative_regrets)
    ### with regularizer 1e-8 instead of 0 we can avoid the if-else branch (see _orig code)


def get_regrets(payoff: int, action_2: Action) -> np.array:
    """return (non-negative) regrets"""
    z = np.array([get_payoff(a, action_2) - payoff for a in Action])
    z = np.maximum(z, 0)        # bug fix compared to rps_regret_match_orig, yields smoother curves and
                                # allows to skip strategy accumulation (i.e. CUMULATE_STRAT = False)
    return z

cumulative_regrets = np.zeros(shape=(len(Action)), dtype=int)
cumulative_opp_regrets = np.zeros(shape=(len(Action)), dtype=int)
strategy_sum = np.zeros(shape=(len(Action)))

fixed_opp_strategy = [0.2, 0.2, 0.6]
#fixed_opp_strategy = np.repeat(1/len(Action),len(Action))

strat_arr = np.zeros(shape=(num_iterations,len(Action)))
our_payoff_arr = np.zeros(shape=(num_iterations))

for k in range(num_iterations):
    #  compute the strategy according to regret matching
    strategy = get_strategy(cumulative_regrets)

    #if k % 100 == 0: print(cumulative_regrets,cumulative_opp_regrets)
    #if k % 100 == 0: print(np.argmax(cumulative_regrets),np.argmax(cumulative_opp_regrets))
    
    if CASE=="INDEPEND":
        opp_strategy = get_strategy(cumulative_opp_regrets)
    elif CASE=="STEAL":
        # opponent 'steals' our strategy: she calculates our cumulative regrets,
        # finds our most probable action and selects her best counter-action
        opp_strategy = np.zeros(shape=(len(Action)))
        opp_strategy[(np.argmax(strategy)+1)%3]=1
        

    if CUMULATE_STRAT:
        #  add the strategy to our running total of strategy probabilities:
        strategy_sum += strategy
        optimal_strategy = strategy_sum/(k+1)
    else:
        # take just the strategy emerging from the cumulative regrets:
        optimal_strategy = strategy
   
    # just for later plotting: 
    # strat_arr is 2D array with actions as columns and k as rows
    strat_arr[k,:] = optimal_strategy
    
    # Choose our action and our opponent's action
    our_action = random.choices(list(Action), weights=strategy)[0]
    if CASE=="FIXED":
        opp_action = random.choices(list(Action), weights=fixed_opp_strategy)[0]
    elif CASE=="SELFPLAY":
        opp_action = random.choices(list(Action), weights=strategy)[0]
    elif CASE=="INDEPEND" or CASE=="STEAL":
        opp_action = random.choices(list(Action), weights=opp_strategy)[0]
    else: 
        print("Unallowed value CASE =",CASE)
        break
    
    #  compute the payoff and regrets
    our_payoff = get_payoff(our_action, opp_action)
    regrets = get_regrets(our_payoff, opp_action)    
    #  add regrets from this round to the cumulative regrets
    cumulative_regrets += regrets
    
    our_payoff_arr[k] = our_payoff
    
    if CASE=="INDEPEND":
        # compute opponent payoff and opponent regret independent from our's
        opp_payoff = get_payoff(opp_action, our_action)
        opp_regrets = get_regrets(opp_payoff, our_action)
        cumulative_opp_regrets += opp_regrets
    
print("Case {}".format(CASE))
print("our optimal strategy: {}".format(optimal_strategy))
print("our average payoff = {}".format(np.mean(our_payoff_arr[2000:])))     # 2000: after transient phase
plt.plot(strat_arr)
plt.show()
#print(strat_arr)