# Imports
import numpy as np
import pandas as pd 
from scipy.stats import linregress
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_elo(teams, data, initial_rating=2000, k=140, alpha=None, weights=False, nan_score=1):
    '''
    Calculate Elo ratings for each team based on match data.

    Parameters:
    - teams (array-like): Containing Team-IDs.
    - data (pd.DataFrame): DataFrame with all matches in chronological order.
    - initial_rating (float): Initial rating of an unranked team (default: 2000).
    - k (float): K-factor, determining the impact of each match on team ratings (default: 140).
    - alpha (float or None): Tuning parameter for the multiplier for the margin of victory. No multiplier if None.

    Returns: 
    - list: Historical ratings of the winning team (WTeam).
    - list: Historical ratings of the losing team (LTeam).
    '''
    
    # Dictionary to keep track of current ratings for each team
    team_dict = {}
    for team in teams:
        team_dict[team] = initial_rating
        
    # Lists to store ratings for each team in each game
    r1, r2 = [], []
    loss = []
    margin_of_victory = 1
    weight = 1

    # Iterate through the game data
    for wteam, lteam, ws, ls, w  in tqdm(zip(data.WTeamID, data.LTeamID, data.WScore, data.LScore, data.weight), total=len(data)):

        # Calculate expected outcomes based on Elo ratings
        rateW = 1 / (1 + 10 ** ((team_dict[lteam] - team_dict[wteam]) / initial_rating))
        rateL = 1 / (1 + 10 ** ((team_dict[wteam] - team_dict[lteam]) / initial_rating))
        
        if alpha:
                margin_of_victory = (ws - ls)/alpha
        if isinstance(weights, (list, np.ndarray, pd.Series)):
            weight = w

        # Update ratings for winning and losing teams
        team_dict[wteam] += w * k * margin_of_victory * (1 - rateW)
        team_dict[lteam] += w * k * margin_of_victory * (0 - rateL)

        # Ensure that ratings do not go below 1
        if team_dict[lteam] < 1:
            team_dict[lteam] = 1
            
        # Append current ratings for teams to lists
        r1.append(team_dict[wteam])
        r2.append(team_dict[lteam])
        loss.append((1-rateW)**2)
        
    return r1, r2, loss

def create_elo_data(teams, data, initial_rating=2000, k=140, alpha=None, weights=None, nan_score=1):
    '''
    Create a DataFrame with summary statistics of Elo ratings for teams based on historical match data.

    Parameters:
    - teams (array-like): Containing Team-IDs.
    - data (pd.DataFrame): DataFrame with all matches in chronological order.
    - initial_rating (float): Initial rating of an unranked team (default: 2000).
    - k (float): K-factor, determining the impact of each match on team ratings (default: 140).
    - weights (array-like): Containing weights for each match.

    Returns: 
    - DataFrame: Summary statistics of Elo ratings for teams throughout a season.
    '''
    
    if isinstance(weights, (list, np.ndarray, pd.Series)):
        data['weight'] = weights
    else:
        data['weight'] = 1
    
    r1, r2, loss = calculate_elo(teams, data, initial_rating, k, alpha, weights, nan_score)
    # Calculate loss only on tourney results
    loss = np.mean(np.array(loss)[data.tourney == 1])
    print(f"Loss: {loss}")
    
    # Concatenate arrays vertically
    seasons = np.concatenate([data.Season, data.Season])
    days = np.concatenate([data.DayNum, data.DayNum])
    teams = np.concatenate([data.WTeamID, data.LTeamID])
    tourney = np.concatenate([data.tourney, data.tourney])
    ratings = np.concatenate([r1, r2])
    # Create a DataFrame
    rating_df = pd.DataFrame({
        'Season': seasons,
        'DayNum': days,
        'TeamID': teams,
        'Rating': ratings,
        'Tourney': tourney
    })

 # Sort DataFrame and remove tournament data
    rating_df.sort_values(['TeamID', 'Season', 'DayNum'], inplace=True)
    rating_df = rating_df[rating_df['Tourney'] == 0]
    grouped = rating_df.groupby(['TeamID', 'Season'])
    results = grouped['Rating'].agg(['mean', 'median', 'std', 'min', 'max', 'last'])
    results.columns = ['Rating_Mean', 'Rating_Median', 'Rating_Std', 'Rating_Min', 'Rating_Max', 'Rating_Last']
    results['Rating_Trend'] = grouped.apply(lambda x: linregress(range(len(x)), x['Rating']).slope, include_groups=False)
    results.reset_index(inplace=True)
    
    return results

# Load and Process Data Men's Tourney
regular_m = pd.read_csv('/Users/zhangzongyu/Desktop/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
tourney_m = pd.read_csv('/Users/zhangzongyu/Desktop/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
teams_m = pd.read_csv('/Users/zhangzongyu/Desktop/march-machine-learning-mania-2025/MTeams.csv')

regular_m['tourney'] = 0
tourney_m['tourney'] = 1
regular_m['weight'] = 1
tourney_m['weight'] = 0.7

data_m = pd.concat([regular_m, tourney_m])
data_m.sort_values(['Season', 'DayNum'], inplace=True)
data_m.reset_index(inplace=True, drop=True)

elo_df_men = create_elo_data(teams_m.TeamID, data_m, initial_rating=1200, k=125, alpha=None, weights=data_m['weight'])
elo_df_men.tail(10)

# Load and Process Data Women's Tourney
regular_w = pd.read_csv('/Users/zhangzongyu/Desktop/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv')
tourney_w = pd.read_csv('/Users/zhangzongyu/Desktop/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')
teams_w = pd.read_csv('/Users/zhangzongyu/Desktop/march-machine-learning-mania-2025/WTeams.csv')

regular_w['tourney'] = 0
tourney_w['tourney'] = 1
regular_w['weight'] = 0.95
tourney_w['weight'] = 1

data_w = pd.concat([regular_w, tourney_w])
data_w.sort_values(['Season', 'DayNum'], inplace=True)
data_w.reset_index(inplace=True, drop=True)

elo_df_women = create_elo_data(teams_w.TeamID, data_w, initial_rating=1250, k=190, alpha=None, weights=data_w['weight'])
elo_df_women.tail(10)