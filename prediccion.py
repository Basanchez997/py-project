#--------------Importar librerias
import tensorflow as tf #linreria de IA
import cv2 #Lib de opencv
import numpy as np #Operaciones
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array



#Direccion de los modelos
ModeloDenso = 'D:/py-lib/project/py-project/ClasificadorDenso.h5'
ModeloCNN = 'D:/py-lib/project/py-project/ClasificadorCNN.h5'
ModeloCNN2 = 'D:/py-lib/project/py-project/ClasificadorCNN2.h5'


#Leemos las redes neuronales
#Denso
Denso = tf.keras.models.load_model(ModeloDenso)
pesosDenso = Denso.get_weights()
Denso.set_weights(pesosDenso)
#CNN
CNN = tf.keras.models.load_model(ModeloCNN)
pesosCNN = CNN.get_weights()
CNN.set_weights(pesosCNN)
#CNN2
CNN2 = tf.keras.models.load_model(ModeloCNN2)
pesosCNN2 = CNN2.get_weights()
CNN2.set_weights(pesosCNN2)

#Realizamos una videocaptura

cap = cv2.VideoCapture(0)
#Empieza nuestro while true

while True:
    #Lectura de nuestra video captura
    ret, frame = cap.read()

    #Pasamos a escala de grises
    gray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Redimensionamos la imagen
    gray = cv2.resize(gray, (200,200), interpolation= cv2.INTER_CUBIC)

    #Normalizamos laimagen
    gray = np.array(gray).astype(float)/255

    #Convertimos la imagen en matriz
    img = img_to_array(gray)
    img = np.expand_dims(img, axis=0)

    #Realizamos la prediccion
    prediccion  = CNN2.predict(img)
    prediccion = prediccion[0]
    prediccion = prediccion[0]
    print(prediccion)

    #Realizamos la clasificacion
    if prediccion <=0.5:
        cv2.putText(frame, "Manzana?", (200, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Naranja?", (200, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)

    #Mostramos los fotogramas
    cv2.imshow("CNN", frame)

    t = cv2.waitKey(1)
    if t == 27:
        break
cv2.destroyAllWindows()
cap.release()











