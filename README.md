# Lane Change Prediction from Highway Trajectories

This repository contains an early-stage research project on lane change prediction using real-world highway trajectory data.

The project focuses on building a basic machine learning pipeline for:
- trajectory preprocessing
- feature construction from driving sequences
- recurrent neural network baselines for lane change prediction

This repository is intended as a research portfolio project. It demonstrates how raw trajectory data can be organized into a reproducible workflow for sequence modeling and experimental comparison.

## Project Overview

The goal of this project is to predict future lane change behavior from historical vehicle trajectories on highways.

The current version explores:
- preprocessing of trajectory data
- sequence-based feature extraction
- recurrent models such as RNN, LSTM, and GRU
- comparison of different model variants for lane change prediction

This is an early-stage project and does not represent a full autonomous driving system.

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── models/
│   ├── core_counting.py
│   ├── data_check.py
│   ├── data_display.py
│   ├── data_splitting.py
│   ├── deepIRL.py
│   ├── dump_data.py
│   ├── general_IRL.py
│   └── ...