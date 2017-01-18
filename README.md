# sound_classification
Data is from Urban sound classification dataset.

# features
The mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC. 

The idea is that MFCC vectors are important features. An audio file can be expressed in the form of frequency vs time diagram. 
Some other features are extracted using librosa.

# classification
The first model is using RandomForestClassifer, which give 69% accuracy. From what I read online, some achieve over 80% accuracy on this dataset. I tried to use Neural Network. However, I ran into this problem. Time series data are inherently difficult for Neural Network. I am putting this project on the backburner until I came up with better idea.
