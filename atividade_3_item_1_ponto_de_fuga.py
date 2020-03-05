import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

#%matplotlib inline

#1. Printando a Versão do OpenCV
print(cv2.__version__)

#2
print(os.getcwd())

#3
proto = "./mobilenet_detection/MobileNetSSD_deploy.prototxt.txt"
model = "./mobilenet_detection/MobileNetSSD_deploy.caffemodel"

#4 Intanciação da rede neural
net = cv2.dnn.readNetFromCaffe(proto, model)

#5 Categorias da MobileNet

#6 Lendo a Imagem
#cap = cv2.VideoCapture("VID_20200302_063445951.mp4")
cap = cv2.VideoCapture(0)
CONFIDENCE = 0.7
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#7 Função Detect

#8 Contador para o Item 2

#9 Lista de Elementos para selecionar

#10 Vídeo
while(True):

# Lendo a imagem
    ret,frame = cap.read()

# Colocando a imagem na função detect
    saida, resultados = detect(frame,achou)

# Printando os resultados no consolse
    print(resultados)

# Contandos os itens na imagem
    for detectecao in resultados:
        print(detectecao[0])
        if detectecao[0] in escolhidos:
            contador +=1
            break
        else:
            contador = 1
            achou = False

# Desenhando os quadrados
    if contador >= 5:
        achou = True
    print(contador)

# Plotando a imagem na tela
    cv2.imshow("Saida", saida)

# Atalho para sair q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()