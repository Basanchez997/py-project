#--------------Importar librerias
from gc import callbacks
import os
from pickletools import optimize
import tensorflow as tf #linreria de IA
import cv2 #Lib de opencv
import matplotlib.pyplot as plt #Lib de observar imagenes
import numpy as np #Operaciones
from tensorflow.keras.callbacks import TensorBoard #Lb para observar el funcionamineto de l ared 
from tensorflow.keras.preprocessing.image import ImageDataGenerator #lib para modificar mis img

#------Direcciones de imagenes-----------
#entrenamiento = r'C:/Users/ERTECH/Desktop/Dataset'
#validacion = r'C:/Users/ERTECH/Desktop/Dataset'

entrenamiento = r'C:/Users/brayd/Desktop/Dataset'
validacion = r'C:/Users/brayd/Desktop/Dataset'

listaTrain = os.listdir(entrenamiento)
listaTest = os.listdir(validacion)

print(listaTrain) 

print(listaTest) 

#---algunos parametros-------
ancho, alto = 200,200
#Lista entrnamiento
etiquetas = []
fotos = []
datos_train = []
con = 0
#Lista validacion
etiquetas2 = []
fotos2 = []
datos_vali = []
con2 = 0

#----Extraer en una lista las fotos y entra las etiquetas
#Entrenamiento
for nameDir in listaTrain:
    nombre = entrenamiento + '/' + nameDir # Leemos las fotos
    print(nombre) 
    for fileName in os.listdir(nombre): #Asignamos las etquetas a cada foto
        etiquetas.append(con) #valor de la etiqueta(asignamios 0 a la primera etiqueda y 1 a la segunda)
        img = cv2.imread(nombre + '/' + fileName, 0) #Leemos la imagen
        img = cv2.resize(img, (ancho,alto), interpolation = cv2.INTER_CUBIC) #Redimensiar imagenes
        img = img.reshape(ancho, alto, 1) #Dejamos 1 solo canal
        datos_train.append([img, con])
        fotos.append(img) #Añadimos las imagenes en EDG

    con = con + 1
#Validacion
for nameDir2 in listaTest:
    nombre2 = validacion + '/' + nameDir2 # Leemos las fotos

    for fileName2 in os.listdir(nombre2): #Asignamos las etquetas a cada foto
        etiquetas2.append(con2) #valor de la etiqueta(asignamios 0 a la primera etiqueda y 1 a la segunda)
        img2 = cv2.imread(nombre2 + '/' + fileName2, 0) #Leemos la imagen
        img2 = cv2.resize(img2, (ancho,alto), interpolation = cv2.INTER_CUBIC) #Redimensiar imagenes
        img2 = img2.reshape(ancho, alto, 1) #Dejamos 1 solo canal
        datos_vali.append([img2, con2])
        fotos2.append(img2) #Añadimos las imagenes en EDG

    con2 = con2 + 1

#------Normañiozar ñas o,amges de (0 o 1)
fotos= np.array(fotos).astype(float)/255
print(fotos.shape)
fotos2 = np.array(fotos2).astype(float)/255
print(fotos.shape)
#Pasamods las lista a Array
etiquetas = np.array(etiquetas)
etiquetas2 = np.array(etiquetas2)

#Sirve para que puede identifica la imagen en tiempo real y no solo las fotos
imgTrainGen = ImageDataGenerator(
    rotation_range = 50, #Rotacion aleatoria de las imagenes
    width_shift_range = 0.3, #Mover la imagen a los lados
    height_shift_range = 0.3, #Mover la imagen arria y abajo
    shear_range= 15, #Inclinamos la imagen
    zoom_range=15, #Hacemos zoom a la imagen
    vertical_flip= True, #Flip verticales aleatorios
    horizontal_flip= True #Flip horizontal aleatorio
)
imgTrainGen.fit(fotos)
plt.figure(figsize=(20,8))
for imagen, etiqueta in imgTrainGen.flow(fotos, etiquetas, batch_size=10, shuffle=False):
    for i in range(10):
        plt.subplot(2, 5, i +1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(imagen[1], cmap='gray')
    plt.show()
    break
imgTrain = imgTrainGen.flow(fotos, etiquetas, batch_size=32)

#Estructura de la red nuronal convulucional
#Modelo con capas densas
ModeloDenso = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (200,200,1)), #Capas de entrata de 40,000 neuraonas
    tf.keras.layers.Dense(150, activation = 'relu'), #Capad ndesa con 150 neurnas
    tf.keras.layers.Dense(150, activation = 'relu'), #Capad ndesa con 150 neuronas
    tf.keras.layers.Dense(1,activation = 'sigmoid'),
])

#Modelo con capas convolucionales
ModeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (200,200, 1)), #Capa entrada convolu
    tf.keras.layers.MaxPooling2D(2,2), #capa de max pooling
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'), #Capa conmvoluional con 64 kernel 
    tf.keras.layers.MaxPooling2D(2,2), #Capa max pooling
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'), #Capa convolucional de 128 kernel
    tf.keras.layers.MaxPooling2D(2,2),

    #Capas densas de clasificacion
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'), #Capa densa con 256
    tf.keras.layers.Dense(1, activation = 'sigmoid')

])

ModeloCNN2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation = 'relu', input_shape = (200,200, 1)), #Capa entrada convolu
    tf.keras.layers.MaxPooling2D(2,2), #capa de max pooling
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'), #Capa conmvoluional con 64 kernel 
    tf.keras.layers.MaxPooling2D(2,2), #Capa max pooling
    tf.keras.layers.Conv2D(128, (3,3), activation = 'relu'), #Capa convolucional de 128 kernel
    tf.keras.layers.MaxPooling2D(2,2),

    #Capas densas de clasificacion
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation = 'relu'), #Capa densa con 256
    tf.keras.layers.Dense(1, activation = 'sigmoid')

])

#cOMPILAMOS LOS MODELOS: AGREGAMOS EL OPTMIZADOR Y LA FUNCION DE PERDIDA
ModeloDenso.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

ModeloCNN.compile(optimizer = 'adam',
                    loss= 'binary_crossentropy',
                    metrics = ['accuracy'])

ModeloCNN2.compile(optimizer = 'adam',
                    loss= 'binary_crossentropy',
                    metrics = ['accuracy'])

#Observaremos y entrenaremos nuestras redes
#para  visualizar: tensordboard --logdir="C:/Users/ERTECH/Desktop/"
#para  visualizar: tensorboard --logdir="C:/Users/brayd/Desktop/Procesos"
#Entrenamos el modelo Denso
#BoardDenso = TensorBoard(log_dir='C:/Users/ERTECH/Desktop/')
BoardDenso = TensorBoard(log_dir='C:/Users/brayd/Desktop/Procesos')
ModeloDenso.fit(imgTrain, batch_size = 32, validation_data = (fotos2,etiquetas2),
                epochs = 100, callbacks = [BoardDenso], steps_per_epoch = int(np.ceil(len(fotos) / float(32))),
                validation_steps = int(np.ceil(len(fotos2) / float(32))))
#Guardamos el modelo
ModeloDenso.save('ClasificadorDenso.h5')
ModeloDenso.save_weights('pesosDenso.h5')
print("Terminamos Modelo denso")

#Entrenamos CNN sin 00
#BoardCNN = TensorBoard(log_dir='C:/Users/ERTECH/Desktop/')
BoardCNN = TensorBoard(log_dir='C:/Users/brayd/Desktop/Procesos')
ModeloCNN.fit(imgTrain, batch_size = 32, validation_data = (fotos2,etiquetas2),
                epochs = 100, callbacks = [BoardCNN], steps_per_epoch = int(np.ceil(len(fotos) / float(32))),
                validation_steps = int(np.ceil(len(fotos2) / float(32))))
#Guardamos el modelo
ModeloCNN.save('ClasificadorCNN.h5')
ModeloCNN.save_weights('pesosCNN.h5')
print("Terminamos Modelo CNN 1")

#Entrenamos CNN con 00
#BoardCNN2 = TensorBoard(log_dir='C:/Users/ERTECH/Desktop/')
BoardCNN2 = TensorBoard(log_dir='C:/Users/brayd/Desktop/Procesos')
ModeloCNN2.fit(imgTrain, batch_size = 32, validation_data = (fotos2,etiquetas2),
                epochs = 100, callbacks = [BoardCNN2], steps_per_epoch = int(np.ceil(len(fotos) / float(32))),
                validation_steps = int(np.ceil(len(fotos2) / float(32))))
#Guardamos el modelo
ModeloCNN.save('ClasificadorCNN2.h5')
ModeloCNN.save_weights('pesosCNN2.h5')
print("Terminamos Modelo CNN 2")

