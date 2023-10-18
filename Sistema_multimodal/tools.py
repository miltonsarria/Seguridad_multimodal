import numpy as np
import pyaudio
from helpAudio import  byte_to_float

########################
######### configurar py audio
#funcion que abre el microfono, graba durante RECORD_SECONDS y 
#retorno la muestras del audio
#CHUNK tamano del buffer
#FORMAT = pyaudio.paInt16
#CHANNELS = cuantos canales?
#RATE  frecuencia de muestreo
#RECORD_SECONDS duracion total en segundos

def getAudio(objPyaudio,
             CHUNK = 1024,
             FORMAT = pyaudio.paInt16,
             CHANNELS = 1,
             RATE = 16000,
             RECORD_SECONDS = 3):
   
    stream = objPyaudio.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

    print("[INFO] Escuchando...") #cuando vea el mensaje debe hablar
    audio = []
    #concatenar los CHUNK que se reciben
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        data = byte_to_float(data)
        audio.append(data)
    print("[INFO] hecho!") #cuando vea el mensaje debe dejar de hablar
    audio =np.hstack(audio)
    #detener y cerrar el stream
    stream.stop_stream()
    stream.close()
   
    return audio        