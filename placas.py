#Importa librerias
import cv2 
import numpy as np #Operaciones
import pytesseract
from PIL import Image

#Video captura
#video="http://192.168.0.107:31522/videostream.cgi?user=admin&pwd=888888" 
#video = 'lento.mp4'
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
    #Tomar el cntro de la imagen en X:
    x1 = int(an / 2) # tomamos el 1/3 de la imagen
    x2 = int(x1 * 2 ) # Hasta el inicio del 3/3 de la imagen

    #En Y:
    y1 = int(al / 2) # tomamos el 1/3 de la imagen
    y2 = int(y1 * 2 ) # Hasta el inicio del 3/3 de la imagen

    #Texto
    cv2.rectangle(redmin, (x1 + 160, y1 + 500), (300, 600), (0, 0, 0), cv2.FILLED)
    cv2.putText(redmin, 'PROCESANDO PLACOTA ', (x1 + 180, y1 + 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #uBICAMOS EL RECTANGULO EN LA ZONAS EXTRAIDAS
    cv2.rectangle(redmin, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #Realizamos un recorte a nuestra zona de interes
    recorte = redmin[y1:y2, x1:x2]
    #prueba
    recorte2 = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)
    #Reprocesamiento de la zona de interes
    mB = np.matrix(recorte[:, :, 0])
    mG = np.matrix(recorte[:, :, 1])
    mR = np.matrix(recorte[:, :, 2])

    #restamos el color verde y azul para sacar amarillo
    Color = cv2.absdiff(mB, mR)
    Color = cv2.absdiff(mB, mR, mG)
    #cv2.imshow("Test", recorte)}
    #prueba
    _, umbral = cv2.threshold(Color, 150, 255, cv2.THRESH_BINARY)
    #_, umbral = cv2.threshold(recorte2, 180, 200, cv2.THRESH_BINARY)
    #cv2.imshow("Antes", umbral)
    #prueba = cv2.blur(umbral,(3,3))
    #umbral_limpio = cv2.Canny(umbral, 150, 200)
    umbral_limpio = cv2.dilate(umbral, None, iterations=1)
    #cv2.imshow("Test", umbral_limpio)
    #Extraemos los contornos de la zona seleccionada
    contornos, _ = cv2.findContours(umbral_limpio, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #Primero los ordenamos del mas grande al mas pequeño
    contornos = sorted(contornos, key=lambda x : cv2.contourArea(x), reverse=True)
    
    #Dibujamos los contornos extraidos
    for contorno in contornos:
        area = cv2.contourArea(contorno)
        #print("areaaa", area)
        if area > 15000:
            #cv2.imshow("Test", area)
            #Detectamos la placa
            x, y, ancho, alto = cv2.boundingRect(contorno)

            #Extraemos las coodernadas
            xpi = x + x1 #Coordenada de la placa en x inicial
            ypi = y + y1 #Coordenada de la placa en Y inicial

            xpf = x + ancho + x1 #Coordenada de la placa en x final
            ypf = y + alto + y1 #Coordenada del aplaca en y final

            #Dibujamos el rectangulo
            cv2.rectangle(redmin, (xpi, ypi), (xpf, ypf), (255, 255, 0), 2)
            
            #Extraemos los pixeles
            placa = redmin[ypi:ypf, xpi:xpf]
            #cv2.drawContours(placa, [contorno], 0, (0,255,0),2)
            cv2.imshow("placa", placa)
            #Extraemos el ancho y el alto de los fotogramas
            alp, anp, cp = placa.shape
            #print(alp, anp)
            
            #Procesamos los piceles para extraer los valores de las palcas
            Mva = np.zeros((alp, anp))

            #Normalizamos las matrices

            mBp = np.matrix(placa[:, :, 0])
            mGp = np.matrix(placa[:, :, 1])
            mRp = np.matrix(placa[:, :, 2])
            
            #Creamos una mascara

            #estamos el color verde y azul para sacar amarillo
            Color2 = cv2.absdiff(mBp, mRp)
            Color2 = cv2.absdiff(mBp, mRp, mGp)
            #cv2.imshow("nose", Color2)
            #Binarizamos l aimagen
            #cv2.imshow("Test", recorte)
            _, umbral2 = cv2.threshold(Color2, 150, 255, cv2.THRESH_BINARY)
            #cv2.imshow("Antes", umbral)
            #prueba = cv2.blur(umbral,(3,3))
            #umbral_limpio2 = cv2.Canny(umbral2, 150, 200)
            umbral_limpio2 = cv2.dilate(umbral2, None, iterations=1)
            #cv2.imshow("Test", umbral_limpio)
            #for col in range(0, alp):
             #   for fil in range(0, anp):
              #      Max = max(mRp[col,fil], mGp[col,fil], mBp[col, fil])
               #     Mva[col, fil] = 255 - Max
                #    print("Max",Max)
                 #   cv2.imshow("Mva", Mva)
            #cv2.imshow("umbral_limpio2", umbral_limpio2)
            #Binarizamos la imagen
            _, bin = cv2.threshold(umbral_limpio2, 150, 255, cv2.THRESH_BINARY)

            #Covertimos la matriz en imagen
            bin = bin.reshape(alp, anp)
            bin = Image.fromarray(bin)
            bin = bin.convert("L")
            #Nos aseguramos de tener un buen tamaño de la placa
            if alp >= 50 and anp >= 100:
                #Declaramos la direccion de Pyresseract
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                #Extraemos el texto
                config = "--psm 1"
                texto = pytesseract.image_to_string(umbral_limpio2, config=config)
                print("aui",texto)
                #If para no mostrar basura
                if len(texto) >= 5:
                    Ctexto = texto
        
        
        
            
            break
            
    #Mostramos el reporte en gris
    #cv2.imshow("Vehiculos", frame)
    
    #cv2.imshow("Vehiculos2", redmin)

    #Leemos una tecla
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()

