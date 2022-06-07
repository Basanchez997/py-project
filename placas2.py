#Importa librerias
import cv2 
import numpy as np #Operaciones
import pytesseract
from PIL import Image

#Video captura
#video="http://192.168.0.107:31522/videostream.cgi?user=admin&pwd=888888" 
#video = 'VIDEOMOTO.mp4'
video = "http://192.168.0.128:4747/video"
#video = 0
cap = cv2.VideoCapture(video)

Ctexto = ''


#Creamos nuestro while true
while True:
#Realizamos la lectura de la videoCaptura
    ret, frame = cap.read()
    al1, an1, c1 = frame.shape
    redmin = cv2.resize(frame, (an1*2, al1*2))
    if ret == False:
        break

    #Dibujamos un rectangulo
    cv2.rectangle(redmin, (870, 750), (1070, 850), cv2.FILLED)
    cv2.putText(redmin, Ctexto[0:7], (900, 810), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #eXTRAEMOS EL ANCHO Y EL ALTO DE LOS FOTOGRAMAS
    al, an, c = redmin.shape
    #cv2.imshow("Test", redmin)
    #Tomar el cntro de la imagen en X:
    x1 = int(an / 2) # tomamos el 1/3 de la imagen
    x2 = int(x1 * 2 ) # Hasta el inicio del 3/3 de la imagen

    #En Y:
    y1 = int(al / 2) # tomamos el 1/3 de la imagen
    y2 = int(y1 * 2 ) # Hasta el inicio del 3/3 de la imagen

    #Texto
    cv2.rectangle(redmin, (x1 + 160, y1 + 500), (1120, 940), (0, 0, 0), cv2.FILLED)
    cv2.putText(redmin, 'PROCESANDO PLACOTA ', (x1 + 180, y1 + 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #uBICAMOS EL RECTANGULO EN LA ZONAS EXTRAIDAS
    cv2.rectangle(redmin, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #Realizamos un recorte a nuestra zona de interes
    recorte = redmin[y1:y2, x1:x2]
    recorte2 = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gris?", recorte2)
    #Reprocesamiento de la zona de interes
    mB = np.matrix(recorte[:, :, 0])
    mG = np.matrix(recorte[:, :, 1])
    mR = np.matrix(recorte[:, :, 2])

    #restamos el color verde y azul para sacar amarillo
    Color = cv2.absdiff(mB, mR)
    Color = cv2.absdiff(mB, mR, mG)
    #cv2.imshow("Antes", Color)
    #Binarizamos l aimagen
    #cv2.imshow("Test", recorte)
    _, umbral = cv2.threshold(Color, 180, 200, cv2.THRESH_BINARY)
    umbral_limpio = cv2.dilate(umbral, None, iterations=2)
    cv2.imshow("Test", umbral_limpio)
    #Extraemos los contornos de la zona seleccionada
    contornos, _ = cv2.findContours(umbral_limpio, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Primero los ordenamos del mas grande al mas pequeño
    contornos = sorted(contornos, key=lambda x : cv2.contourArea(x), reverse=True)
    
    t = cv2.waitKey(1)

    if t == 27:
        break
        cap.release()
        cv2.destroyAllWindows()

