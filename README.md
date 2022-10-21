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
    * [Advance Visualisation](#advance-visualisations)

* [Conclusion](#conclusion)
* [Authors](#authors)

<!--te-->

# Introduction

The National Hockey League (NHL; French: Ligue nationale de hockey—LNH) is a professional ice hockey league in North
America comprising 32 teams—25 in the United States and 7 in Canada. It is considered to be the top ranked professional
ice hockey league in the world, and is one of the major professional sports leagues in the United States and Canada.
The Stanley Cup, the oldest professional sports trophy in North America,is awarded annually to the league playoff
champion at the end of each season. The NHL is the fifth-wealthiest professional sport league in the world by revenue,
after the National Football League (NFL), Major League Baseball (MLB), the National Basketball Association (NBA), and
the English Premier League [EPL](https://en.wikipedia.org/wiki/National_Hockey_League)

## Motivation

The purpose of this project is to provide a Python API for accessing NHL game data including plays by plays
informations such as game summaries, player stats and play-by-play visualizations. They’re all lots good information
that is hides on NHL API website scraping process regarding the outputs. In this project we are trying to show all NHL
analytics we could like to seek from NHL API. Our package can extract let’s say all most the game summary report as
well as show and finally permit advanced data visualisations.

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

In this project as of now we have used two APIs which was provided by NHL,

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

     ```
     TODO
     Discuss how you could add the actual strength information (i.e. 5 on 4, etc.) to both shots and goals, given the 
     other event types (beyond just shots and goals) and features available.
     Ans: 

     In a few sentences, discuss at least 3 additional features you could consider creating from the data available in 
     this dataset. We’re not looking for any particular answers, but if you need some inspiration, could a shot or 
     goal be classified as a rebound/shot off the rush (explain how you’d determine these!)
    
    - We can classify a shot as a rebound shot or not based on the timing on the previous shot. 
    - Based on the given even shot of the player, how likely the shot can be a goal
    - 
    
     ```

<details>
<summary>Tidy Data</summary>
     <h4>Insights</h4>
     There is too much information available from the NHL API at this moment. Not all information are useful, based 
     on the project we take the relevant data out from the nested json and create a single tabular structure aka
     Dataframe. Below is a glimpse of the tidy data which we had published for further data analysis.

<img src="figures/df.png">
</details>

## Interactive Debugging Tool

<details>
<summary>Goals By Season for the season 2020</summary>
     <h4>Insights</h4>
     To be added here. 
     <img src="figures/idt.png"/>
</details>

## Simple Visualisations

<details>
<summary>Goals By Season for the season 2016</summary>
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
<summary>Goals By Season for the season 2018</summary>
     <h4>Insights</h4>
     The proportion of goals over all shots increases overall exponentially as the distance diminishes, with a
     maximum proportion of goals >25% when goals are shot at less than 5 feet from the goal. We also note a small,
     local maximum at 75 to 80 feet. This distribution did not increase significantly for seasons 2018-19 to 2020-21. This local
     maximum could suggest that there is another variable (e.g. shot type or other) that could underlie this
     distribution.
     <br>
     <br>
     We chose this figure after having considered and visualized different types of figures. First, we visualized
     violin plots of the distribution of goals and missed shots; however, these did not intuitively represent the chance
     (proportion) of goals over all shots per se, and the result was dependent on some assumption on the kernel size.
     We also experimented computing a logistic regression to predict goals from the distance category, which worked fine.
     <br>
     <br>
     Finally, we chose to come back to the most simple and intuitive method, which is to bin the distance into categories,
     and plot the proportion of goals for each bin. We chose to divide the distance into equal bins (as opposed to
     percentiles or other kind of distribution), in order to be able to draw direct conclusion about the relationship of
     goals to the absolute value of distance by visualizing the figure.
<img src="figures/figure_2_goal_by_distance2018.png"/>
</details>

<details>
<summary>Goals By Season for the season 2019</summary>
     <h4>Insights</h4>
     Overall, the most dangerous type of shot is the “tip-in” shot taken at a distance of less than 5 feet, followed
     closely by “back-hand” shots: more than 40% of these shots result in a goal. The relationship found in the
     previous questions, i.e. that the probability of a goal augments exponentially as the distance decreases,
     holds true overall for most types of shots. However, the “deflected” and “tip-in” shots have a second maximum
     between around 30 and 60 feet.
     <br>
     <br>
     Importantly, the “back-hand” shot has a second maximum at about 80 feet, and the slap-shot has a second maximum
     at more than 90 feet. This could explain the small local maximum at that distance that we observed in the global
     distribution of all shots at the previous figure.
     <br>
     <br>
     Finally, the curves are somewhat irregular, and adding more data (e.g. averaging through a few years) could add more
     smoothness in the results. Note that to have more smoothed curves and remove outliers, we did not plot the points for
     which we had less than 10 total observations for that type of shot and at that distance in that season.
<img src="figures/figure_2_goal_by_distance2019.png"/>
</details>

<details>
<summary>Goals By Season for the season 2020</summary>
     <h4>Insights</h4>
     To be added here. 
     <img src="figures/figure_2_goal_by_distance2020.png"/>
</details>

<details>
<summary>Goals By Distance and Shot type for the season 2017</summary>
     <h4>Insights</h4>
     To be added here. 
     <img src="figures/figure_3_goals_by_distance_and_shot_type2017.png">
</details>

## Advance Visualisations

<details>
<summary>Season 2016-2017 Comparison </summary>
     <h4>Insights</h4>
     To be added here. 
     <img src="figures/figure_3_goals_by_distance_and_shot_type2017.png" width="300">
     <img src="figures/figure_3_goals_by_distance_and_shot_type2017.png" width="300"> 
</details>

# Conclusion

To be added

# Authors

- ABC
- DEF
- GHI
- JKL


