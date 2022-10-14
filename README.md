# Hockey Primer Data Exploration

![](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/Markdown-000000?style=for-the-badge&logo=markdown&logoColor=white)
![](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)
![](http://ForTheBadge.com/images/badges/made-with-python.svg)
![Version](https://img.shields.io/badge/Version-1.0.0-green)


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

## About NHL

## Motivation

# Installation

## Setting up Environment
- Git clone the [repository](git@github.com:amandalmia14/hockey-primer-1.git)
- Make sure Python is installed on the system
- Create a virtual environment / conda environment

## Installing Dependencies
- Activate the environment and run `pip install -r requirement.txt`

## Download Data
- The data for the NHL games are exposed out in the form of various APIs, the details of the APIs can be found over 
[here](https://gitlab.com/dword4/nhlapi)
- Run the python script which resides at `modules/data_extraction.py`, this script will fetches the data of the seasons
starting from 2016 to 2020. 



# Project Structure

As seen the above image, the project is divided into various parts, 
- `data` - It contains all the NHL tournament data season wise, in each season we have two json files of regular season 
game and playoffs. 
- `modules` - For every actions which we are performing in this project, are been captured as modules, like data 
extractions, data retrival (data parsing)
- 'notebooks' - For all kinds of visualisations, insights of the data can be accessed through the notebooks. 
- 'constants.py' - As name suggested, all the common functions and variables reside in this file.

# Data Insights

## Data Extractions

## Interactive Debugging Tool

## Simple Visualisations

## Advance Visualisations 

# Conclusion

# Authors



