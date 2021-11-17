import librosa 
import librosa.display 
import numpy as np

def loadTrainData(fpathA,fpathO,nAnswer,nOther):
    path = fpathA
    file_exten = '.m4a'
    num_files = nAnswer
    answer = np.ones((num_files,1))
    arr = np.zeros((0,0))

    for num in range(1,num_files+1):
        fname = path + str(num) + file_exten
        y, sr = librosa.load(fname)
        #stfft with window size of 32768 2^15
        S = np.abs(librosa.stft(y,32768))
        S = np.mean(S,axis=1)
        norm = np.linalg.norm(S)
        S = S/norm
        if num == 1:
            arr = S
        else:
            arr = np.vstack((arr,S))

    path = fpathO
    file_exten = '.m4a'
    num_files = nOther
    answer = np.vstack((answer,np.zeros((num_files,1))))
    for num in range(1,num_files+1):
        fname = path + str(num) + file_exten
        y, sr = librosa.load(fname)

        #stfft with window size of 32768 2^15 1024
        S = np.abs(librosa.stft(y,32768))
        S = np.mean(S,axis=1)
        norm = np.linalg.norm(S)
        S = S/norm
        arr = np.vstack((arr,S))

    answer.astype("float32")

    np.savetxt('audioClickData',arr)
    np.savetxt('audioClickAnswers',answer)
    test = np.loadtxt('audioClickData')
    print(test.shape)

def loadTestData(fpathA,fpathO,nAnswer,nOther):
    path = fpathA
    file_exten = '.m4a'
    num_files = nAnswer
    answer = np.ones((num_files,1))
    arr = np.zeros((0,0))

    for num in range(1,num_files+1):
        fname = path + str(num) + file_exten
        y, sr = librosa.load(fname)
        #stfft with window size of 32768 2^15
        S = np.abs(librosa.stft(y,32768))
        S = np.mean(S,axis=1)
        norm = np.linalg.norm(S)
        S = S/norm
        # S = S.flatten()
        if num == 1:
            arr = S
        else:
            arr = np.vstack((arr,S))

    path = fpathO
    file_exten = '.m4a'
    num_files = nOther
    answer = np.vstack((answer,np.zeros((num_files,1))))
    for num in range(1,num_files+1):
        fname = path + str(num) + file_exten
        y, sr = librosa.load(fname)
        #stfft with window size of 32768 2^15 1024
        S = np.abs(librosa.stft(y,32768))
        S = np.mean(S,axis=1)
        norm = np.linalg.norm(S)
        S = S/norm
        # S = S.flatten()
        arr = np.vstack((arr,S))

    answer.astype("float32")

    np.savetxt('audioClickTestData',arr)
    np.savetxt('audioClickTestAnswers',answer)


