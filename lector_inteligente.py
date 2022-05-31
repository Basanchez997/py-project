#Importa librerias
import cv2 
import pytesseract
from gtts import gTTS
#from playsound import playsound

#Inicializar variables
cuadro = 100
anchocam , altocam = 640, 480

video="http://192.168.0.107:31522/videostream.cgi?user=admin&pwd=888888" 
#captura del video
cap = cv2.VideoCapture(video)
cap.set(3,anchocam) #Definiremos alto y ancho de la camara
cap.set(4,altocam)

#Funcion para leer texto

#While principal




def text(image):
    #Funcio para reproducir la voz
    def voz(archi_text, lenguaje, nom_archi):
        with open(archi_text, "r") as lec: #Abrimos el archivo de texto
            lectura = lec.read()
        lect = gTTS(text = lectura, lang = lenguaje, slow= False)
        nombre = nom_archi
        lect.save(nombre)
        
        
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texto = pytesseract.image_to_string(gris)
    print(texto)
    dire = open('Info.txt', "w")
    dire.write(texto)
    dire.close()
   


while True:
    ret, frame = cap.read()#leemos la captura y almacenamos los fpsen frame
    if ret == False:break #si no puede conectar con la camara o video se detnga,
    cv2.putText(frame, 'Ubique el texto a leer', (158, 80), cv2.FONT_HERSHEY_PLAIN, 0.71, (255, 255, 0), 2)
    cv2.putText(frame, 'Ubique el texto a leer', (160, 80), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 2)
    cv2.rectangle(frame, (cuadro,cuadro), (anchocam - cuadro, altocam - cuadro), (0, 0, 0), 2) # Genramos cuadro
    x1, y1 = cuadro, cuadro  #Extraemos las coordenadas de la esquiza superior izquierda
    ancho, alto = (anchocam - cuadro) - x1, (altocam - cuadro) - y1 #Extraemos el ancho y el alto
    x2, y2 = x1 + ancho, y1 + alto #Almacenamos los pixiles el recuadro
    doc = frame[y1:y2, x1:x2]
    cv2.imwrite("Imatxt.jpg", doc)

    cv2.imshow("Lector inteligente", frame)
    t = cv2.waitKey(1)
    if t == 27:
        break



text(doc)
cap.release()
cv2.destroyAllWindows()









