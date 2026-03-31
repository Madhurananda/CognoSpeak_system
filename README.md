# CognoSpeak_system
This repository contains codes which are used to run standard and foundation models on acoustic and linguistic features extracted from 126 participants who are either healthy or suffer from Dementia or MCI. 

The results are published in the paper titled "CognoSpeak: an automatic, remote assessment of early cognitive decline in real-world conversational speech" which is accepted for presentation at the 2025 IEEE Symposium on Computational Intelligence in Health and Medicine. 

- requirements.txt and requirements.yml files describe the Anaconda packages used in this project. 

## Please cite our paper as:

@inproceedings{pahar2025cognospeak,
  title={Cogno{S}peak: an automatic, remote assessment of early cognitive decline in real-world conversational speech},
  author={Pahar, Madhurananda and Tao, Fuxiang and Mirheidari, Bahman and Pevy, Nathan and Bright, Rebecca and Gadgil, Swapnil and Sproson, Lise and Braun, Dorota and Illingworth, Caitlin and Blackburn, Daniel and Christensen, Heidi},
  booktitle={2025 IEEE Symposium on Computational Intelligence in Health and Medicine (CIHM)}, 
  year={2025},
  volume={},
  number={},
  pages={1-7},
  doi={10.1109/CIHM64979.2025.10969487}
}

Pahar, M., Tao, F., Mirheidari, B., Pevy, N., Bright, R., Gadgil, S., Sproson, L., Braun, D., Illingworth, C., Blackburn, D. and Christensen, H., 2025, March. CognoSpeak: an automatic, remote assessment of early cognitive decline in real-world conversational speech. In 2025 IEEE Symposium on Computational Intelligence in Health and Medicine (CIHM) (pp. 1-7). IEEE.


## Run the scripts: 
### Acoustic Classifiers
python CognoSpeak_acoustics.py <int (number of CPU)> |& tee -a ../logs/acoustic_results.txt


### Linguistic Classifiers

python CognoSpeak_linguistics.py <str (token name)> <list (list of GPUs)> |& tee -a ../logs/linguistics_results.txt

#### Here, the token name is "CognoSpeak" and four GPUs are to be used parallelly whose cuda IDs are 0, 1, 2 and 3 
python CognoSpeak_linguistics.py CognoSpeak 0,1,2,3 |& tee -a ../logs/linguistics_results.txt

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14515541.svg)](https://doi.org/10.5281/zenodo.14515541)
