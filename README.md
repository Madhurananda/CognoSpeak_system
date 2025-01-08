# CognoSpeak_system
This repository contains codes which are used to run standard and foundation models on acoustic and linguistic features extracted from 126 participants who are either healthy or suffer from Dementia or MCI. 

The results are published in the paper titled "CognoSpeak: an automatic, remote assessment of early cognitive decline in real-world conversational speech" which is accepted for presentation at the 2025 IEEE Symposium on Computational Intelligence in Health and Medicine. 

- requirements.txt and requirements.yml files describe the Anaconda packages used in this project. 

## Please cite our paper as:
- To be updated 

## Run the scripts: 
### Acoustic Classifiers
python CognoSpeak_acoustics.py <int (number of CPU)> |& tee -a ../logs/acoustic_results.txt


### Linguistic Classifiers

python CognoSpeak_linguistics.py <str (token name)> <list (list of GPUs)> |& tee -a ../logs/linguistics_results.txt

#### Here, the token name is "CognoSpeak" and four GPUs are to be used parallelly whose cuda IDs are 0, 1, 2 and 3 
python CognoSpeak_linguistics.py CognoSpeak 0,1,2,3 |& tee -a ../logs/linguistics_results.txt

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14515541.svg)](https://doi.org/10.5281/zenodo.14515541)
