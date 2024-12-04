# metis

## Description
Metis is named after the Greek Oceanid symbol of wisdom and deep thought. In mythology, Metis was known for her intelligence and cunning, and she played a key role as the mother of Athena, the goddess of wisdom. This repository draws inspiration from her legacy, aiming to bring thoughtful and innovative contributions to problem of data-driven symbolic regression.

## Table of Contents
- [Installation](#installation)
- [Folder Structure](#folder-structure)

## Installation
The conda environment related to this work, can be installed using:
```bash
  conda env create -f metis_env.yaml
```
or, if one uses the mamba package manager:
```bash
  mamba env create -f metis_env.yaml
```

# Folder Structure
In the `src` folder you can find the `datagen` class, which can be used to generate systems of random differential equations, also in their symbolic form, and also to generate trajectories for each equation in the system.