from keras.models import Sequential
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.layers import Activation,Flatten,Dense,Dropout
from keras.preprocessing.image import ImageDataGenerator

training_data="training data path"
validation_data="validation data path"

img_width,img_height=150,150

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen=ImageDataGenerator(rescale=1./255)

#Data augmentation for improving the model (not used here)
train_datagen_augmented = ImageDataGenerator(
        rescale=1./255,        # normalize pixel values to [0,1]
        shear_range=0.2,       # randomly applies shearing transformation
        zoom_range=0.2,        # randomly applies shearing transformation
        horizontal_flip=True)  # randomly flip the images

# automagically retrieve images and their classes for train and validation sets

training_data_gen=datagen.flow_from_directory(training_data,
                                              target_size=(img_width,img_height),
                                              batch_size=16,
                                              class_mode="binary")
validation_data_gen=datagen.flow_from_directory(validation_data,
                                                target_size=(img_width,img_height),
                                                batch_size=32,
                                                class_mode="binary")




def createmodel():

    model=Sequential()
    model.add(Convolution2D(32,3,3,input_shape=(img_width,img_height,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(32, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation("relu"))

    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    return model

model=createmodel()

model.compile(loss="binary_crossentropy",
              optimizer='rmsprop',
              metrics=['accuracy'])

#Training
epochs=30
train_samples = 2048
validation_samples = 832

model.fit_generator(training_data_gen,
                    samples_per_epoch=train_samples,
                    nb_epoch=epochs,
                    validation_data=validation_data_gen,
                    nb_val_samples=validation_samples)

#Evaluating on validation set

model.evaluate_generator(validation_data_gen,validation_samples)

#Save the model
modelJSON=model.to_json()
with open('C:/Users/Madhu/Desktop/DogsVsCats.json','w') as jsonfile:
    jsonfile.write(modelJSON)

model.save_weights('C:/Users/Madhu/Desktop/DogsVsCats.h5')