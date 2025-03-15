import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import glob
import warnings
warnings.filterwarnings("ignore")

path = '/Users/zhangzongyu/Desktop/march-machine-learning-mania-2025/*.csv'
data = {p.split('/')[-1].split('.')[0]: pd.read_csv(p) for p in glob.glob(path)}

# Feature Engineering

# Combine men's and women's team data into one DataFrame
teams = pd.concat([data['MTeams'], data['WTeams']])

# Combine men's and women's team spelling data
teams_spelling = pd.concat([data['MTeamSpellings'], data['WTeamSpellings']])

# Count different spellings for each team
teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()

# Rename columns to be more descriptive
teams_spelling.columns = ['TeamID', 'TeamNameCount']

# Add spelling count to teams DataFrame
teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])

# Combine men's and women's regular season compact results
season_cresults = pd.concat([data['MRegularSeasonCompactResults'], data['WRegularSeasonCompactResults']])

# Combine men's and women's regular season detailed results
season_dresults = pd.concat([data['MRegularSeasonDetailedResults'], data['WRegularSeasonDetailedResults']])

# Combine men's and women's tournament compact results
tourney_cresults = pd.concat([data['MNCAATourneyCompactResults'], data['WNCAATourneyCompactResults']])

# Combine men's and women's tournament detailed results
tourney_dresults = pd.concat([data['MNCAATourneyDetailedResults'], data['WNCAATourneyDetailedResults']])

# Combine men's and women's tournament slot data
slots = pd.concat([data['MNCAATourneySlots'], data['WNCAATourneySlots']])

# Combine men's and women's tournament seed data
seeds = pd.concat([data['MNCAATourneySeeds'], data['WNCAATourneySeeds']])

# Combine men's and women's game cities data
gcities = pd.concat([data['MGameCities'], data['WGameCities']])

# Combine men's and women's seasons data
seasons = pd.concat([data['MSeasons'], data['WSeasons']])

# Create dictionary mapping season_teamID to seed number (extract numeric part of seed)
seeds = {'_'.join(map(str,[int(k1),k2])):int(v[1:3]) for k1, v, k2 in seeds[['Season', 'Seed', 'TeamID']].values}

# Get cities data
cities = data['Cities']

# Get submission template
sub = data['SampleSubmissionStage2']

# Add column to mark regular season compact results as 'S' (Season)
season_cresults['ST'] = 'S'

# Add column to mark regular season detailed results as 'S' (Season)
season_dresults['ST'] = 'S'

# Add column to mark tournament compact results as 'T' (Tournament)
tourney_cresults['ST'] = 'T'

# Add column to mark tournament detailed results as 'T' (Tournament)
tourney_dresults['ST'] = 'T'

# Combine all detailed results into one DataFrame
games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)

# Reset index of combined games
games.reset_index(drop=True, inplace=True)

# Convert win location to numeric (Away=1, Home=2, Neutral=3)
games['WLoc'] = games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

# Create unique ID for each game with season and sorted team IDs
games['ID'] = games.apply(lambda r: '_'.join(map(str, [r['Season']]+sorted([r['WTeamID'],r['LTeamID']]))), axis=1)

# Create ID for team matchup (sorted team IDs)
games['IDTeams'] = games.apply(lambda r: '_'.join(map(str, sorted([r['WTeamID'],r['LTeamID']]))), axis=1)

# Get lower team ID as Team1
games['Team1'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[0], axis=1)

# Get higher team ID as Team2
games['Team2'] = games.apply(lambda r: sorted([r['WTeamID'],r['LTeamID']])[1], axis=1)

# Create ID for Season_Team1
games['IDTeam1'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)

# Create ID for Season_Team2
games['IDTeam2'] = games.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

# Map Team1's seed for that season, fill missing with 0
games['Team1Seed'] = games['IDTeam1'].map(seeds).fillna(0)

# Map Team2's seed for that season, fill missing with 0
games['Team2Seed'] = games['IDTeam2'].map(seeds).fillna(0)

# Calculate score difference (winner - loser)
games['ScoreDiff'] = games['WScore'] - games['LScore']

# 1 if Team1 won, 0 if Team2 won (target variable)
games['Pred'] = games.apply(lambda r: 1. if sorted([r['WTeamID'],r['LTeamID']])[0]==r['WTeamID'] else 0., axis=1)

# Normalize score diff from Team1 perspective
games['ScoreDiffNorm'] = games.apply(lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0. else r['ScoreDiff'], axis=1)

# Calculate seed difference (Team1 - Team2)
games['SeedDiff'] = games['Team1Seed'] - games['Team2Seed']

# Fill missing values with -1
games = games.fillna(-1)

# List of statistical columns to aggregate (game stats)
c_score_col = ['NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
 'LBlk', 'LPF']

# List of aggregation functions to apply
c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std', 'skew', 'nunique']

# Group by team matchups and calculate historical stat aggregates
gb = games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()

# Rename aggregated columns with suffix
gb.columns = [''.join(c) + '_c_score' for c in gb.columns]

# Filter to tournament games only (we'll train only on tournament data)
games = games[games['ST']=='T']

# Set all submission games to neutral location (3)
sub['WLoc'] = 3

# Extract season from ID
sub['Season'] = sub['ID'].map(lambda x: x.split('_')[0])

# Convert season to integer
sub['Season'] = sub['Season'].astype(int)

# Extract Team1 from ID
sub['Team1'] = sub['ID'].map(lambda x: x.split('_')[1])

# Extract Team2 from ID
sub['Team2'] = sub['ID'].map(lambda x: x.split('_')[2])

# Create team matchup ID
sub['IDTeams'] = sub.apply(lambda r: '_'.join(map(str, [r['Team1'], r['Team2']])), axis=1)

# Create Season_Team1 ID
sub['IDTeam1'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1)

# Create Season_Team2 ID
sub['IDTeam2'] = sub.apply(lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1)

# Map Team1's seed, fill missing with 0
sub['Team1Seed'] = sub['IDTeam1'].map(seeds).fillna(0)

# Map Team2's seed, fill missing with 0
sub['Team2Seed'] = sub['IDTeam2'].map(seeds).fillna(0)

# Calculate seed difference
sub['SeedDiff'] = sub['Team1Seed'] - sub['Team2Seed']

# Fill missing values with -1
sub = sub.fillna(-1)

# Add aggregated stats to tournament games
games = pd.merge(games, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')

# Add aggregated stats to submission template
sub = pd.merge(sub, gb, how='left', left_on='IDTeams', right_on='IDTeams_c_score')

# Select feature columns, excluding IDs and raw stats
col = [c for c in games.columns if c not in ['ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2',
                                             'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff', 'ScoreDiffNorm',
                                             'WLoc'] + c_score_col]

# model

# Selecting training data
X = games[col]
sub_X = sub[col]

# XGB parameters
param_grid = {
    'n_estimators': 5000,
    'learning_rate': 0.03,
    'max_depth': 6,
    'device': 'cpu',
    'random_state': 42
}

# Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('xgb', XGBRegressor(**param_grid))
])

# Fitting pipeline
pipeline.fit(X, games['Pred'])

# Predicting games and submissions
pred = pipeline.predict(X).clip(0.001, 0.999)
sub_pred = pipeline.predict(sub_X).clip(0.001, 0.999)

# Results

# Cross validation (for the MSE)
cv_scores = cross_val_score(pipeline, X, games['Pred'], cv=5, scoring="neg_mean_squared_error")

# Results
print(f'Log Loss: {log_loss(games["Pred"], pred):.5f}')
print(f'Mean Absolute Error: {mean_absolute_error(games["Pred"], pred):.5f}')
print(f'Brier Score: {brier_score_loss(games["Pred"], pred):.5f}')
print(f'Cross-validated MSE: {-cv_scores.mean():.5f}')