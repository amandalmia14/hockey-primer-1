# Hockey Primer Data Exploration

![](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)
![](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)
![](https://img.shields.io/badge/Version-1.0.0-green)


Table of contents
=================

<!--ts-->

* [Introduction](#introduction)
    * [Motivation](#motivation)
* [Installation](#installation)
    * [Setup Environment](#setup-environment)
    * [Install Dependencies](#install-dependencies)
* [Usage](#usage)
    * [Download Data](#download-data)
    * [Run Interactive Debugging Tool](#run-interactive-debugging-tool)
    * [Create Tidy Data for Visualisation](#create-tidy-data-for-visualisation)
    * [Run Simple Visualisation](#run-simple-visualisation)
    * [Run Advance Visualisation](#run-advance-visualisation)
* [Project Structure](#project-structure)
* [Data APIs](#data-apis)
* [Data Insights](#data-insights)
    * [Data Extractions](#data-extractions)
    * [Interactive Debugging Tool](#interactive-debugging-tool)
    * [Simple Visualisation](#simple-visualisations)
    * [Advanced Visualisation](#advanced-visualisations)

* [Conclusion](#conclusion)
* [Authors](#authors)

<!--te-->

# Introduction

The National Hockey League (NHL) is a professional ice hockey league in North America. It comprises a total of 32 teams, of which 7 in Canada [EPL](https://en.wikipedia.org/wiki/National_Hockey_League), including the Montreal Canadien. Each year, the Stanley Cup playoffs selects the best team, which is awarded the Stanley Cup for the season. For example, The Montreal Canadien won the Stanley Cup 24 for 24 seasons, the last time in 1992-1993. 

The NHL makes publically available an API that features statistics including meta-data on each season, season standings, player statistics by season, and play-by-play data. This last format of data is the most thorough and it features important information for all events during each game, such as the players involved, location coordinates on the ice, and the type of event. The NHL API is a valuable source of fine-grained sports data that can be used in a number of tasks such as finding the features that predict goals, or those that predict players salaries. 


## Motivation

The purpose of this project is to provide a Python API for accessing NHL data, specifically, all the play-by-play informations. The reader will learn here how to download the NHL data for a given year, how to first visualize it, and then how to format it into a tidy data frame. This tidy data format will then be used for producing simple, as well as more advanced, interactive visualizations. At term, this data could also be used for a number of purpuses including machine learning, or other tasks at the reader's will. 

# Installation

## Setup Environment

- Git clone the [repository](https://github.com/amandalmia14/hockey-primer-1)
- Make sure Python is installed on the system
- Create a virtual environment / conda environment

## Install Dependencies

- Activate the environment and run `pip install -r requirement.txt` this will download all the dependencies related to
  this project.

# Usage

## Download Data
- The data for the NHL games are exposed out in the form of various APIs, the details of the APIs can be found over
  [here](https://gitlab.com/dword4/nhlapi)
- Run the python script which resides at `modules/dataextraction/data_extraction.py`, this script will fetch the data 
of the seasons starting from 2016 to 2020.
- This will create a folder in your directory for the season which you want to download and two json files will be 
appeared along with some other files which will be used later part of the project.  
  - `YYYY_regular_season.json`
  - `YYYY_playoffs.json`
  
  <br>
  <img src="figures/data_download.png" width="200"/>
  <br>

## Run Interactive Debugging Tool
- Run the `jupyter notebook` locally inside the project folder
- Navigate to the `notebook` folder 
- Run `3_interactive_debugging_tool.ipynb` file

## Create Tidy Data for Visualisation
- Run the python script which resides at `modules/dataretrival/data_retrival.py`, this script will creates the tidy data 
and save the data into a pickle file for all the seasons starting from 2016 to 2020.

## Run Simple Visualisation
- Run the `jupyter notebook` locally inside the project folder (Incase if jupyter notebook isn't running)
- Navigate to the `notebook` folder 
- Run `4_simple_visualizations.ipynb` file

## Run Advance Visualisation
- Run the `jupyter notebook` locally inside the project folder (Incase if jupyter notebook isn't running)
- Navigate to the `notebook` folder 
- Run `7_interactive_figure.ipynb` file

# Project Structure

As seen in the above image, the project is divided into various parts,

- `data` - It contains all the NHL tournament data season wise, in each season we have two json files of regular season
  games and playoffs.
- `figures` - It contains all the data insights which we captured in this project. 
- `modules` - For every action which we are performing in this project, are captured as modules, like data
  extractions, data retrieval (data parsing)
- `notebooks` - For all kinds of visualisations, insights of the data can be accessed through the notebooks.
- `constants.py` - As the name suggests, all the common functions and variables reside in this file.

# Data APIs

This project uses two APIs which were provided by the NHL : 

- `GET_ALL_MATCHES_FOR_A_GIVEN_SEASON = "https://statsapi.web.nhl.com/api/v1/schedule?season=XXXX"`
    - This API fetch all the matches metadata for a given input season, using this API we are getting the map of
      Matches ID and the type of Match it is like `regular season or playoffs`
- `GET_ALL_DATA_FOR_A_GIVEN_MATCH = "https://statsapi.web.nhl.com/api/v1/game/XXXXXXXXXX/feed/live/"`
    - This API fetches all the data in a granular form for a given match id, where we gather the insights subsequently
      in
      the following tasks.
- In order to download a particular data for a season, update the file `modules\dataextraction\data_extraction.py` with
  the `year` variable (one can put multiple seasons to download as well)
- Once the update is done, run `data_extraction.py` it will download the data and place it under a folder with the
  season
  year with two json files, with regular season games and playoffs respectively.

# Data Insights

## Data Extractions

The data available by the NHL API needs to be parsed and formatted in order to make more advanced data usages possible. In this regard, we select the relevant data out from the nested dictionnaries from the json file, and and we format a single tabular structure, i.e. 
 a dataframe. Below is a glimpse of the tidy dataframe which will be used in further analyses. 

<details>
<summary>Tidy Data</summary>
<img src="figures/df.png">
</details>

<h3>How to get the number of players in each team</h3> 
The first step would be to format a new tidy dataframe which would includes all types of events (not only the shots and goals, such as in the dataframe featured above), with events as rows and including datetime, eventType, periodType, penaltySeverity, penaltyMinutes, and team, as columns. The events would to be sorted in order of their occurrence in time during the game (datetime).

We would then create an empty (np.nan) column for the number of players on ice, and program a loop to iterate over all event, while concatenating a list of player counts for each time, n_1 and n_2. At the beginning of the loop, and at the beginning of each period
(each time the period of the event is not the same as the previous event), we re-initiate the parameters: n_1 = 6 (number of players in first team, including the goalie), n_2 = 6 (number of players in second team, including the goalie).

Eight parameters would be set: penalty_player_A_team_1=None, end_time_of_penalty_for_player_A_team_1 = Datetime penalty_player_B_team_1 = None, and end_time_of_penalty_for_player_B_team_1=Datetime (as there can be a maximum of 2 players in penalty at the same time);and the four equivalent parameters for team 2. 

Then, as the loop iterater over all events, each time the eventTypeId == "PENALTY", if "penaltySeverity": "Minor" or "DoubleMinor", the number of player in the team involved in the penalty (Team of the player that is penalized) would be substracted 1, the penalty_player would be set to the name the penalized player, and end_time_of_penalty parameter would be set to DateTime + penaltyMinutes. For subsequent events, as long as the penalty_player is not None, the datetime of the event would be compared to end_time_of_penalty, untill datetime > end_time_of_penalty and then the number of player for that team would be added +1, as the player is back on ice.

Note that for other types of penalty (e.g. misconduct), the number of player on the ice would not be updated as an immediate player replacement is allowed.

<h3>Engineering additionnal features </h3>
We would be interested in studying the impact of tackling and hitting on the chance of goals, both (1) at team-level (2 variables), (2) player-level (4 variables), and (3) total through the game (4 variables). Indeed, tackling and hitting has become an important part of hockey, often discussed by commentators, and highly represented in the data under "eventTypeId": "HIT". 

(1) We would first extract, for each shot event, variables at team-level that corresponds to the time (in minutes) between the shot and the last time a player of the team on which the shot was taken was hit. This would be done by iterating through all events in chronological order, initiating the time at as NaN at the beginning of each period, and updating the time at each time a hit happens, for each team. This would result in variables: time_since_last_hit_team_1 and time_since_last_hit_team_2. 

(2) Additionally, during the same iteration process, we would update four boolean variables with player-level information to note whether the hitter and the hittee from the last hit event were among the player involved in the shot (shooter, goalie or assist). This would result in variables: hitter_involved_team_1, hittee_involved_team_1, hitter_involved_team_2, hittee_involved_team_2. 

(3) Finally, to study the relationship between goals and the total number of hits in a game, we would extract 4 variables, during the same iteration process as above. These variables would be initiated at 0 at the beginning of the game, and updated at each hit event for each team and type of player involved (hitter or hittee). This would result in variables: n_hitter_team_1, n_hittee_team_1, n_hitter_team_2, n_hittee_team_2.

## Interactive Debugging Tool

<details>
<summary>Goals By Season for the season 2020</summary>
     <h4>Insights</h4>
     To be added here. 
     <img src="figures/idt.png"/>
</details>

## Simple Visualisations

<details>
<summary>Goals and missed shots, by shot type for season 2016-2017</summary>
     <h4>Insights</h4>
     The most dangerous types of shots for this 2016-2017 season are “deflected” (19.8% of success) followed by 
     “tip-in” shots (17.9% of success). By “most dangerous”, we mean that these shots are the ones that end up the most 
     frequently by a successful goal, as opposed to being missed. However, these are among the less frequent ones: 
     there were only 806 “deflected” and 3,267 “tip-in” shots this season. On the contrary, the most common type of 
     shots was by far the “wrist shot”, with a total of 38,056 shots of that type for this season.
     <br>
     <br>
     We chose to illustrate this question with a barplot while overlaying the count of goals in blue overtop the total 
     count of shots in orange (thus, total of both goals and other, missed shots), by type of shot. Even though there 
     is a large difference between the most common and less common types of shots, we chose to plot the absolute numbers
     and to keep the scale linear, because these are the most intuitive for the reader to understand the scale 
     (the great number of goals involved in a same season) and not to be confused with too many percentages on the same 
     figure. We chose to add annotations on top of bars for the percentage of goals over all shots, because 
     these proportions could not be visually abstracted simply from the figure, and this was an intuitive way to 
     illustrate them.
     <img src="figures/figure_1_goals_by_shot_type_2016.png"/>
</details>

<details>
<summary>Proportion of goal by distance for seasons 2018, 2019, 2020</summary>
     <h4>Insights</h4>
     The proportion of goals over all shots increases overall exponentially as the distance diminishes, with a maximum 
     proportion of goals >25% when goals are shot at less than 5 feet from the goal. We also note a small, local maximum
     at 75 to 80 feet. This distribution did not increase significantly for seasons 2018-19 to 2020-21. This local 
     maximum could suggest that there is another variable (e.g. shot type or other) that could underlie this 
     distribution. 
     <br>
     <img src="figures/figure_2_goal_by_distance2020.png"/>
</details>

<details>
<summary>Proportion of goal by distance and shot type</summary>
     <h4>Insights</h4>
     We chose this figure after having considered and visualized different types of figures. First, we visualized 
     violin plots of the distribution of goals and missed shots; however, these did not intuitively represent the 
     chance (proportion) of goals over all shots per se, and the result was dependent on some assumption on the kernel 
     size. We also experimented computing a logistic regression to predict goals from the distance category, which 
     worked fine.
     <br>
     Finally, we chose to come back to the most simple and intuitive method, which is to bin the distance into 
     categories, and plot the proportion of goals for each bin. We chose to divide the distance into equal bins (as 
     opposed to percentiles or other kind of distribution), in order to be able to draw direct conclusion about the 
     relationship of goals to the absolute value of distance by visualizing the figure. Overall, the most dangerous \
     type of shot is the “tip-in” shot taken at a distance of less than 5 feet, followed closely by “back-hand” 
     shots: more than 40% of these shots result in a goal. The relationship found in the previous questions, i.e. that 
     the probability of a goal augments exponentially as the distance decreases, holds true overall for most types of 
     shots. However, the “deflected” and “tip-in” shots have a second maximum between around 30 and 60 feet.
     <br>
     Importantly, the “back-hand” shot has a second maximum at about 80 feet, and the slap-shot has a second maximum at 
     more than 90 feet. This could explain the small local maximum at that distance that we observed in the global 
     distribution of all shots at the previous figure.
     <br>
     Finally, the curves are somewhat irregular, and adding more data (e.g. averaging through a few years) could add 
     more smoothness in the results. Note that to have more smoothed curves and remove outliers which made interpretations difficult, we did not plot the 
     points for which we had less than 10 total observations for that type of shot and at that distance in that season. 
     <br>
     <img src="figures/figure_3_goals_by_distance_and_shot_type2017.png">
</details>

## Advanced Visualisations

<details>
<summary>Comparison of Colorado Avalanche avg shots between season 2016-2017 and 2020-2021  </summary>
     <h4>Insights</h4>
     To be added here. 
     <br>
     <img src="figures/Colorado_Avalanche_2016.png" width="325">
     <img src="figures/Colorado_Avalanche_2020.png" width="325"> 
</details>
<details>
<summary>Shots location comparison over the last three years in between Buffalo Sabres and Tampa Bay Lightning</summary>
     <h4>Insights</h4>
     Season 2018-2019 
     <br>
     <img src="figures/Buffalo_Sabres_2018.png" width="325">
     <img src="figures/Tampa_Bay_2018.png" width="325">
     <br>
     <h4>Insights</h4>
     Season 2018-2019 
     <br>
     <img src="figures/Buffalo_Sabres_2019.png" width="325">
     <img src="figures/Tampa_Bay_2019.png" width="325">
     <br>
     <h4>Insights</h4>
     Season 2018-2019 
     <br>
     <img src="figures/Buffalo_Sabres_2020.png" width="325">
     <img src="figures/Tampa_Bay_2020.png" width="325">
     <br>
</details>

# Conclusion

The present project is a good example of how sports data can be obtained from a publically available API, and made into a data format that can be used for advanced, interactive visualizations. Some limitations of the present work include that the conclusions drawn here from the data are based solely on data visualizations, and not yet on thorough predictive modelling. Further work could focus on using the formatted data for tasks such as feature selection and machine learning. 

# Authors

**Vaibhav Jade:** First year student of MSc. Computer Science at UdeM. 

**Mohsen Dehghani:** Master’s degree in Optimization 2010-2013 and student of DESS in Machin learning at MILA 2022-2023. I start a master’s degree in Machin learning at MILA 2022-2023 love to show how to apply theoretical mathematical knowledge to real-life problems by using computer languages such as Java or Python.

**Aman Dalmia:** First year student of MSc. Computer Science at UdeM, have an interest in Information Retrieval and Natural Language Processing. <br>
  *“Don’t create a problem statement for an available solution, rather work towards a solution for a given problem”*

**Raphaël Bourque:** Graduated from Medicine, presently doing residency training in Psychiatry and at the Clinician Investigator Program, and studying at the MSc in Computationnal Medicine. My current research work is in Genetics (CHU Sainte-Justine), and I am very interested in the applications of data science and machine learning to evidence-based medical practice.  

*(Names are in ascending order)*

