# Mila Application: Coding Experience

This public repository is intended to provide an overview of code samples that I have developped throughout my undergraduate studies. Note however that the most important coding projects on which I contributed during my summer research internships are not publicly available for legal reasons.


## Machine Learning

Coding assignments in the graduate class INF8245E Machine Learning at Polytechnique Montreal:

[ML_assignment_1.ipynb](ML_assignment_1.ipynb) (scored 147.5/147.5): Ridge regression, stochastic gradient descent, hyper-parameter optimization with K-Fold cross-validation (manual implementations). The machine learning problem is to predict the power production of a solar power plant as a function of time and weather.

[ML_assignment_2.ipynb](ML_assignment_2.ipynb) (scored 82.0/82.0): KNN, Gaussian Naive Bayes, Logistic Regression classifiers (manual implementations). In particular, we were asked to derive the gradient of cross entropy loss function and implement its gradient descent. The machine learning problem is image classification.

[ML_assignment_3.ipynb](ML_assignment_3.ipynb) (score pending): Naive Bayes, Decision Trees, Logistic Regression, Support Vector Machine classifiers (scikit-learn implementations). Text preprocessing (manual implementations). The machine learning problem is text classification.

[ML_competition.ipynb](ML_competition.ipynb): Machine learning pipeline (feature design, algorithms, methodology, results) to classify concert experiences given a complex dataset. From my explorations, the approach that worked the best was fully-connected neural networks (pytorch implementation).

[ML_acoustic_wave_localization.ipynb](ML_acoustic_wave_localization.ipynb): Personal machine learning project on which I work sporadically on my free time. The idea is to build a touchscreen by localization of finger impacts using machine learning methods. A finger impact on a surface generates a unique set of harmonics (acoustic waves) and neural networks can learn to infer the localization of the impact given the vibrations measurement. This project was inspired from [Huang et al.][1] in which they perform source localization in a water environment. I explore convolutional and recurrent neural networks to leverage time correlation between the features of acoustic waves.


## Engineering

As an engineering undergraduate student, the purpose of most software programs I developped was to interface with hardware devices. Among other things, I coded programs (using Linux SDK provided by the manufacturers) to [probe_electrical_signals.py](probe_electrical_signals.py), to [record_audio_signals.py](record_audio_signals.py) and to [acquire_images.py](acquire_images.py). The live applications required multithreading to handle multiple process simultaneously. Although these code samples are not closely related to machine learning, I judged them relevant to the matter since online learning applications require live data transmission from various sensors.


## Maths / Theory

Beside coding skills, machine learning also requires strong mathematical background. Majoring in Engineering Physics, I had the opportunity to develop skills in theoretical work. This repository contains a [self-review assignment](probs_algebra_hw.pdf) on probability theory and linear algebra, and a [theoretical quantum optics homework](quantum_hw.pdf) (quantum theory framework is essentially a subset of linear algebra, the latest being pertinent to machine learning) for one to assess my capabilities in mathematical formulation of problems.


## Others

In November 2021, I participated in McGill Physics Hackathon [[repo](https://github.com/frmar440/mcgill-physics-hackathon-2021)]. Our team simulated a Stern-Gerlach apparatus. My contribution was to code a complex graph network using [networkx](https://networkx.org/) that would compute dynamically the output probabilities of sending a spin 1/2 particle (of arbitrary orientation) in the system.

In January 2022, I participated in iQuHACK (MIT Quantum Hackathon) [[repo](https://github.com/frmar440/2022_qutech_challenge)]. Our team implemented a quantum key distribution protocol operating under the presence of noise. My contribution was to design the quantum circuit of our protocol and code a program to verify Bell's theorem, and thus verify if the communication has been compromised by an eavesdropper.


[1]: https://asa.scitation.org/doi/10.1121/1.5036725