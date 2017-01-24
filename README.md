# sound_classification
Data is from Urban sound classification dataset.

# features
The mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. Mel-frequency cepstral coefficients (MFCCs) are coefficients that collectively make up an MFC. 

Chroma and tonnetz are two other features.

The idea is that MFCC vectors are important features. An audio file can be expressed in the form of frequency vs time diagram. 
Some other features are extracted using librosa.

# classification
I use RandomForestClassifer model that gets 69% accuracy. 
