import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
from datetime import datetime
import time


def tau(x, y, lamb, mu, rho):
    """
    This function weights the dependance between home xG and away xG, makes it more probable to get evenly matches, low
    scoring games. Will not be implementing this directly, since I am not sure how this works with decimals, also
    makes the partial derivatives waaaaay easier to deal with.

    :param x: goals scored by home team
    :param y: goals scored by away team
    :param lamb: Home team attacking strength * Away team defensive strength * HFA (Home factor advantage)
    :param mu: Home Team defensive strength * Away team attacking strength
    :param rho: Shift parameter, increases likelihood of low scoring games
    :return: dependance factor
    """
    if x == y == 0:
        return 1 - lamb * mu * rho
    elif x == 0 and y == 1:
        return 1 + lamb * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == y == 1:
        return 1 - rho
    else:
        return 1


def decay(md, t=0.0065):
    """
    :param time: Date of result
    :param t: Decay rate, new results should matter more.
    :return:
    """
    match_date = datetime.strptime(md, "%Y-%m-%d")
    return np.exp(-t * ((datetime.today() - match_date).days / 3.5))


def match_log_likelihood(x, y, ai, aj, bi, bj, gamma, rho, match_date):
    lamb = ai*bj*gamma
    mu = aj*bi

    return decay(match_date)*(np.log(tau(x, y, lamb, mu, rho)) - lamb + x*np.log(lamb) - mu + np.log(mu))


def log_likelihood(match_data, parameters):
    log_l = 0

    # Remember gamma and rho are constants for all games
    gamma = parameters[0]
    rho = parameters[1]

    team_parameters = parameters[2]

    # Now, we iterate over every game in the data set and add the log likelihood of this particular event to the
    # total log likelihood function.
    for k in range(len(match_data)):
        match = match_data.iloc[k]

        home_team = match['H']
        away_team = match['A']

        x = match['xG']
        y = match['xGA']
        t = match['Date']

        ai = team_parameters[home_team]['a']
        aj = team_parameters[away_team]['a']

        bi = team_parameters[home_team]['b']
        bj = team_parameters[away_team]['b']

        log_l += match_log_likelihood(x, y, ai, aj, bi, bj, gamma, rho, t)

    return log_l


"""
-------------------------------------------------------------------------------------------------------------------
Below we have functions for partial derivatives
"""


# 1. Partial derivatives with respect to alpha home.
def pd_alpha_home_not_zero(ai, bj, gamma, x, t):
    """
    :return: Partial derivative of Log(L) with respect to ALPHA HOME when result is not low scoring.
    """
    return decay(t)*(x/ai - bj*gamma)


def pd_alpha_home(ai, aj, bi, bj, gamma, rho, t, x, y):
    """
    :return: The partial derivative of Log(L) with respect to ALPHA HOME
    """
    return pd_alpha_home_not_zero(ai, bj, gamma, x, t)


# 2. Partial derivatives with respect to alpha away.
def pd_alpha_away_not_zero(aj, bi, y, t):
    """
    :return: Partial derivative of Log(L) with respect to ALPHA AWAY when result is not low scoring.
    """
    return decay(t)*(y/aj-bi)


def pd_alpha_away(ai, aj, bi, bj, gamma, rho, t, x, y):
    """
    :return: The partial derivative of Log(L) with respect to ALPHA AWAY
    """
    return pd_alpha_away_not_zero(aj, bi, y, t)


# 3. Partial derivatives with respect to beta home.
def pd_beta_home_not_zero(aj, bi, y, t):
    """
    :return: Partial derivative of Log(L) with respect to BETA HOME when result is not low scoring.
    """
    return decay(t)*(y/bi-aj)


def pd_beta_home(ai, aj, bi, bj, gamma, rho, t, x, y):
    """
    :return: The partial derivative of Log(L) with respect to BETA HOME
    """
    return pd_beta_home_not_zero(aj, bi, y, t)


# 4. Partial derivatives with respect to beta away.
def pd_beta_away_not_zero(ai, bj, gamma, x, t):
    """
   :return: Partial derivative of Log(L) with respect to BETA AWAY when result is not low scoring.
    """
    return decay(t)*(x/bj-ai*gamma)


def pd_beta_away(ai, aj, bi, bj, gamma, rho, t, x, y):
    """
    :return: The partial derivative of Log(L) with respect to BETA AWAY
    """
    return pd_beta_away_not_zero(ai, bj, gamma, x, t)


# 5. Partial derivatives with respect to gamma.
def pd_gamma_not_zero(ai, aj, bi, bj, gamma, rho, t, x, y):
    """
    :return: The partial derivative of Log(L) with respect to GAMMA
    """
    return decay(t)*(x/gamma-ai*bj)


"""
------------------------------------------------------------------------------------------------------------------
Now we need to use our partial derivatives to maximize our log likelihood function.
"""
# Maximizing the Log Likelihood function:


def add_to_gradient(match, parameters):
    match_grad = []

    # Extracting relevant outcome data
    x = match['xG']
    y = match['xGA']
    t = match['Date']
    home_team = match['H']
    away_team = match['A']

    # Extracting relevant parameters to calculate partial derivatives.
    gamma = parameters[0]
    rho = parameters[1]
    ai = parameters[2][home_team]['a']
    aj = parameters[2][away_team]['a']
    bi = parameters[2][home_team]['b']
    bj = parameters[2][away_team]['b']

    match_grad.append(pd_gamma_not_zero(ai, aj, bi, bj, gamma, rho, t, x, y))
    # match_grad.append(pd_rho(...))
    match_grad.append(0)


    match_grad.append({})
    # Home team partial derivatives:
    match_grad[2][home_team] = {
                               'a': pd_alpha_home(ai, aj, bi, bj, gamma, rho, t, x, y),
                                'b': pd_beta_home(ai, aj, bi, bj, gamma, rho, t, x, y)
                            }

    # Away team partial derivatives:
    match_grad[2][away_team] =  {
                               'a': pd_alpha_away(ai, aj, bi, bj, gamma, rho, t, x, y),
                               'b': pd_beta_away(ai, aj, bi, bj, gamma, rho, t, x, y)
                           }

    return match_grad


def find_gradient_vector(match_data, parameters):
    gradient_vector = [0, 0, {}]

    for team in parameters[2]:
        gradient_vector[2][team] = {'pd_a': 0, 'pd_b': 0}

    # Every match will add to the gradient by the LL's partial derivatives.
    for index, row in match_data.iterrows():
        gradient_add = add_to_gradient(row, parameters)

        gradient_vector[0] += gradient_add[0]
        gradient_vector[1] += gradient_add[1]

        for team in gradient_add[2]:
            gradient_vector[2][team]['pd_a'] += gradient_add[2][team]['a']
            gradient_vector[2][team]['pd_b'] += gradient_add[2][team]['b']

        # Get the same result for gamma and alpha home on first iteration here.

    return gradient_vector


def maximize(match_data, max_steps=300, learning_rate=0.01):
    """
    This method aims to maximize the log likelihood function and give us the parameters that best fit our Po-model.

    :param match_data: The data we want to maximize our log likelihood  function from.
    :param max_steps: Maximum steps we take in our gradient ascent.
    :param learning_rate: Pretty much step size.
    :return:
    """
    """
        PSEUDO-CODE

        Initialize the parameters randomly

        steps = 0
        while min(grad(log(L))) > epsilon and steps < max_steps:
            for param in vector:
                param += learning_rate*partial_derivative of that param
            steps += 1
        """
    # Initializing the parameters
    parameters = [1, 0.1, {}]
    for team in match_data['H']:
        parameters[2][team] = {'a': 1, 'b': 1}
    for team in match_data['A']:
        parameters[2][team] = {'a': 1, 'b': 1}

    step_count = 0
    while step_count < max_steps:
        grad_vector = find_gradient_vector(match_data, parameters)

        # parameters += learning_rate * grad_vector
        parameters[0] += learning_rate * grad_vector[0]
        parameters[1] += learning_rate * grad_vector[1]

        for team in grad_vector[2]:
            parameters[2][team]['a'] += learning_rate * grad_vector[2][team]['pd_a']
            parameters[2][team]['b'] += learning_rate * grad_vector[2][team]['pd_b']


        #if grad < epsilon
            #break

        step_count += 1
    return parameters


match_logs = pd.read_csv('data.csv')
df = pd.DataFrame(match_logs)
print(match_logs.sort_values('Date', ascending=False))

result = maximize(df)
print(result)
