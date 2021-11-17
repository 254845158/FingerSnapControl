import utils
import tensorflow.keras as keras
import data 

data.loadTestData('../Audio_NN/audio/testClick','../Audio_NN/audio/testOther',4,4)
data.loadTrainData('../Audio_NN/audio/click', '../Audio_NN/audio/other', 9, 33)

#get data and labels
x, y, info = utils.getClickData()
testX, testY, info = utils.getTestData()

#create input tensor with corrosponding size
input = keras.Input((info['sampleSize']))

#create layers and model
dense = keras.layers.Dense(128, activation='relu')
x = dense(input)

x = keras.layers.Dropout(0.4)(x)

x = keras.layers.Dense(64, activation="relu")(x)

x = keras.layers.Dropout(0.4)(x)

output = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs=input, outputs=output, name="test")
print(input.shape)
print(output.shape)
model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)

#train model
history = model.fit(x, y, batch_size=2, epochs=20, shuffle=True)

#eval model
test_scores = model.evaluate(testX, testY, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

model.save("secondAudioModel")
