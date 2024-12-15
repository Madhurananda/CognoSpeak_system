# CognoSpeak_system
This repository contains codes which are used to run standard and foundation models on acoustic and linguistic features extracted from 126 participants who are either healthy or suffer from Dementia or MCI. 

The results are published in the paper titled "CognoSpeak: an automatic, remote assessment of early cognitive decline in real-world conversational speech" which is accepted for presentation at the 2025 IEEE Symposium on Computational Intelligence in Health and Medicine. 

## Run the scripts: 
### Acoustic Classifiers
python CognoSpeak_acoustics.py <int(number of CPU)> |& tee -a ../logs/acoustic_results.txt


### Linguistic Classifiers

python CognoSpeak_linguistics.py <token name> <list of GPUs> |& tee -a ../logs/linguistics_results.txt

#### Here, the token name is "CognoSpeak" and four GPUs to be used parallely whose cuda ids are 0, 1, 2 and 3 
python CognoSpeak_linguistics.py CognoSpeak 0,1,2,3 |& tee -a ../logs/linguistics_results.txt






