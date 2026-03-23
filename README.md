# Replication---Luxuan-Liu-Zhongyuan-Zhou
**Dynamic Coupon Targeting Using Batch Deep Reinforcement Learning**:

This repository contains the data and code used for the replication of the following article:

Liu, X. (2023).
Dynamic coupon targeting using batch deep reinforcement learning: An application to livestream shopping.
Marketing Science, 42(4), 637–658.


**Authors of the replication**:
Luxuan Liu
& Zhongyuan Zhou

School of Hotel and Tourism Management
CUHK Business School



**Repository Structure**:

The data and code are organised into five main folders:

1. Model Free Evidence
2. Model
3. Doubly Robust
4. Field Experiment
5. Data

A brief description of each folder is provided below.



**1. Model Free Evidence**

The folder “Model Free Evidence” contains the code used to generate all figures and tables in the paper that relate to model-free empirical evidence.

To identify which scripts produce specific outputs, please refer to:

Readme_FigureTableSource.xlsx

This file provides a detailed mapping between figures/tables in the paper and the corresponding code files.


**2. Model**

As reported in Table 7 of the paper, six different models were implemented.
Each model folder contains the corresponding training and prediction code.

The models include:

Linear
GBDT
DNN
ORF
Structural Search
Structural McCall (1970)
Structural GBDT
Structural No Search
BDRL

**3. Doubly Robust**

The doubly robust estimate of the new policy value can be obtained by running:

main_live_working_log.py


**4. Field Experiment**

The field experiment was conducted on the Alibaba platform using the reinforcement learning framework A3gentNew, which is a proprietary repository.

An open-source implementation related to this framework can be found at:

https://alibabapai.github.io/easy_rl_doc/using-easyrl-on-pai.html


**5. Data**

The dataset used in this replication is generated through simulation.

To create a representative dataset, run:

simulate_data.R

A small example dataset is provided:

simulated_data.txt

This simulated dataset allows users to run and test the code without access to the original proprietary data.
