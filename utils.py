import numpy as np
import librosa 

def getClickData():
    audioData = np.loadtxt('audioClickData')
    audioLabel = np.loadtxt('audioClickAnswers')
    return audioData, audioLabel, {'sampleSize':audioData.shape[1]}

def getTestData():
    audioData = np.loadtxt('audioClickTestData')
    audioLabel = np.loadtxt('audioClickTestAnswers')
    return audioData, audioLabel, {'sampleSize':audioData.shape[1]}

def preprocess(rawAudio):
    S = np.abs(librosa.stft(rawAudio,32768))
    S = np.mean(S,axis=1)
    norm = np.linalg.norm(S)
    S = S/norm
    return np.reshape(S,(1,S.size))