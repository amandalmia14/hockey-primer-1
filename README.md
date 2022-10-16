# Hockey Primer Data Exploration

![](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)
![](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)
![](https://img.shields.io/badge/Version-1.0.0-green)


Table of contents
=================

<!--ts-->
   * [Introduction](#introduction)
     * [About](#about)
     * [Motivation](#motivation)
   * [Installation](#installation)
      * [Setup Environment](#setup-environment)
      * [Install Dependencies](#install-dependencies)
      * [Download Data](#download-data)
   * [Project Structure](#project-structure)

   * [Data Insights](#dependency)
     * [Data Extractions](#docker)
     * [Interactive Debugging Tool](#docker)
     * [Simple Visualisation](#local)
     * [Advance Visualisation](#public)

   * [Conclusion](#conclusion)
   * [Authors](#authors)
<!--te-->

# Introduction

To be updated

## About NHL

To be updated

## Motivation

To be updated

# Installation

## Setting up Environment
- Git clone the [repository](git@github.com:amandalmia14/hockey-primer-1.git)
- Make sure Python is installed on the system
- Create a virtual environment / conda environment

## Installing Dependencies
- Activate the environment and run `pip install -r requirement.txt` this will download all the dependencies related to 
this project. 

## Download Data
- The data for the NHL games are exposed out in the form of various APIs, the details of the APIs can be found over 
[here](https://gitlab.com/dword4/nhlapi)
- Run the python script which resides at `modules/data_extraction.py`, this script will fetch the data of the seasons
starting from 2016 to 2020. 


# Project Structure

As seen in the above image, the project is divided into various parts, 
- `data` - It contains all the NHL tournament data season wise, in each season we have two json files of regular season 
games and playoffs. 
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
   - This API fetches all the data in a granular form for a given match id, where we gather the insights subsequently in 
   the following tasks.
 - In order to download a particular data for a season, update the file `modules\dataextraction\data_extraction.py` with
the `year` variable (one can put multiple seasons to download as well)
 - Once the update is done, run `data_extraction.py` it will download the data and place it under a folder with the season
year with two json files, with regular season games and playoffs respectively. 

## Data Retrieval

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
     on the project we take the relevant data out from the nested json and created a single tabular structure aka
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
     To be added here.   
     <img src="figures/figure_1_goals_by_shot_type_2016.png"/>
</details>

<details>
<summary>Goals By Season for the season 2018</summary>
     <h4>Insights</h4>
     To be added here. 
     <img src="figures/figure_2_goal_by_distance2018.png"/>
</details>

<details>
<summary>Goals By Season for the season 2019</summary>
     <h4>Insights</h4>
     To be added here. 
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
     <img src="figures/figure_3_goals_by_distance_and_shot_type2017.png"/>
</details>

## Advance Visualisations 

<details>
<summary>Details</summary>
     <h4>Insights</h4>
     To be added here. 
     <img src="figures/figure_3_goals_by_distance_and_shot_type2017.png"/>
</details>

# Conclusion

To be added

# Authors
- ABC 
- DEF
- GHI
- JKL


