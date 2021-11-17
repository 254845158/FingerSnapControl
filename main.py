import sounddevice as sd
import utils
from tensorflow.keras import models
from pynput.keyboard import Key, Controller
from appscript import app, mactypes

#set the desktop to this pic
app('Finder').desktop_picture.set(mactypes.File('./autumn.jpg'))
ctl = True

#load neural network model
model = models.load_model('secondAudioModel')

print("recording...")

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    #pre-process data 
    y = utils.preprocess(indata[:,0])

    #make a prediction with the algo
    pred = model.predict(y)
    global ctl
    if pred > 0.5:
        #some action to carry out
        print("you just snapped you fingers!")
        # keyboard = Controller()
        # with keyboard.pressed(Key.cmd):  
        #     keyboard.tap('w')
        if ctl == True:
            app('Finder').desktop_picture.set(mactypes.File('./autumn_disperse.jpg'))
            ctl = False
        else:
            app('Finder').desktop_picture.set(mactypes.File('./autumn.jpg'))
            ctl = True
    else:    
        print("you didn't snap")
    
    #mute output stream
    outdata.fill(0)

#start audio steam
with sd.Stream(samplerate=22050, channels=1, callback=callback, blocksize=22050):
    print("type anything to stop")
    x = input()
