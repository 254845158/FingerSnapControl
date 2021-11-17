import sounddevice as sd
import utils
from tensorflow.keras import models
from pynput.keyboard import Key, Controller
from appscript import app, mactypes
app('Finder').desktop_picture.set(mactypes.File('./autumn.jpg'))
ctl = True
model = models.load_model('secondAudioModel')

print("recording...")

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    y = utils.preprocess(indata[:,0])
    pred = model.predict(y)
    global ctl
    if pred > 0.5:
        print("you just snapped you fingers!")
        # keyboard = Controller()
        # with keyboard.pressed(Key.cmd):  
        #     keyboard.tap('w')
        # if ctl == True:
        #     app('Finder').desktop_picture.set(mactypes.File('./autumn_disperse.jpg'))
        #     ctl = False
        # else:
        #     app('Finder').desktop_picture.set(mactypes.File('./autumn.jpg'))
        #     ctl = True
    else:    
        outdata.fill(0)
        print("you didn't snap")

#11025 22050
with sd.Stream(samplerate=22050, channels=1, callback=callback, blocksize=22050):
    # sd.sleep(int(duration * 1000))
    print("type anything to stop")
    x = input()
