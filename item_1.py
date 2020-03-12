#importações do Vídeo
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

#Importações para a Máscara
import auxiliar as aux
import math

# If you want to open a video, just change v2.VideoCapture(0) from 0 to the filename, just like below
cap = cv2.VideoCapture('VID_20200302_063445951.mp4')

HEIGHT = cap.read()[1].shape[0]
WIDTH = cap.read()[1].shape[1]
print("Video width:",WIDTH)
print("Video height:",HEIGHT)

pi = np.pi

lower = 0
upper = 1
font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_PLAIN

print("Press q to QUIT")
colors = {
    "vermelho":(0,0,255),
    "azul":(255,0,0)
}
def auto_canny(image, sigma=0.33):
    """apply automatic Canny edge detection using the computed median"""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def coeficientes( array ):
    """Devolve uma tupla (m,n). m é o coeficiente angular e n é o coeficiente linear 
    da reta que representa o conjunto de retas do array"""
    if len(array) > 0:
        # array[:,t] -> coluna t da matriz chamada array
        x1, y1, x2, y2 = array[:,0], array[:,1], array[:,2], array[:,3]
        # m1 = (y2 - y1)/(x2-x1)
        m = (y2 - y1)/(x2 - x1)
        # n1 = m1*x1 + y1
        n = -m*x1 + y1
        
        m = round(np.median(m),2)
        n = round(np.median(n),2)
        return m,n
def tangente(linha):
    "Devolve a tangente do ângulo formado entre a linha e a horizontal (no referencial da imagem)"
    x1,y1,x2,y2 = linha[0]
    return (y2-y1)/(x2-x1)

def draw_line(img, linha, color:str):
    color = colors[color]
    x1,y1,x2,y2 = linha[0]
    cv2.line(img, (x1, y1), (x2, y2), color, 2) 

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Conversão para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redução de ruído 1
    gray = cv2.GaussianBlur(gray,(5,5), 0.33)

    # limiarização da imagem ( fonte: https://youtu.be/P2R7Nn1_VwQ )
    limiar,img_limiar = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Redução de ruido 2
    # Aplicação de um kernel personalizado (deve existir forma de remover a repetição)
    KERNEL = np.array([
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,0,1,0,0],
        [0,0,1,0,0]
    ], dtype="uint8")
    
    segmentado = cv2.morphologyEx(img_limiar,cv2.MORPH_OPEN, KERNEL)
    segmentado = cv2.morphologyEx(segmentado,cv2.MORPH_OPEN, KERNEL)
    segmentado = cv2.morphologyEx(segmentado,cv2.MORPH_OPEN, KERNEL)
    segmentado = cv2.morphologyEx(segmentado,cv2.MORPH_OPEN, KERNEL)
    segmentado = cv2.morphologyEx(segmentado,cv2.MORPH_CLOSE, KERNEL)
    segmentado = cv2.morphologyEx(segmentado,cv2.MORPH_CLOSE, KERNEL)
    segmentado = cv2.morphologyEx(segmentado,cv2.MORPH_CLOSE, KERNEL)
    segmentado = cv2.morphologyEx(segmentado,cv2.MORPH_CLOSE, KERNEL)

    # Redução de ruído 3
    blur = cv2.GaussianBlur(segmentado,(11,11),0.33)

    # Detecção de contornos
    shapes = cv2.Canny(blur, 50, 200)

    # A partir dos contornos, gera um array com linhas que satisfazem os argumentos 
    # minLineLength e maxLineGap (parâmetros foram modificados até que um resultado
    # satisfatório fosse atingido)
    lines = cv2.HoughLinesP(shapes, 1, pi/180, 50, minLineLength=100, maxLineGap=20)
    
    OUTPUT = frame
    
    red_lines = []
    blue_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            try:
                # filtrar as retas por inclinação
                if 0.7 < tangente(line) < 1.3: 
                    red_lines.append(line[0])
                    draw_line(OUTPUT, line, "vermelho") # desenha linhas vermelhas
                elif -1.3 < tangente(line) < -0.7: 
                    blue_lines.append(line[0])
                    draw_line(OUTPUT, line, "azul") # desenha linhas azuis
            except:
                pass
    red_lines = np.array(red_lines)
    blue_lines = np.array(blue_lines)
    try:
        # função 'coeficientes' definida lá em cima
        # m e n são conficientes de retas da forma:  y = mx + n
        m1,n1 = coeficientes( red_lines  )
        m2,n2 = coeficientes( blue_lines )
    except:
        pass
    finally:
        # min() e max() servem para deixar o circulo sempre visível
        # X e Y são as coordenadas do ponto de encontro entre a reta média azul e a reta média vermelha
        X = max(0, min( int((n2-n1)/(m1-m2)), WIDTH ) ) 
        Y = max(0, min( int((m1*X + n1)), HEIGHT ) ) 
        # print("(X,Y) =",(X,Y)) # descomente essa linha para printar no terminal 
        # as coordenadas do centro
        cv2.circle(OUTPUT, (X,Y), 10, (0,255,0), 2, 2)
    
    
    # Adicionar textos na tela
    cv2.putText(OUTPUT," Aperte q para sair", (0,50), font, 1,(255,255,255),2,cv2.LINE_AA)

    # Exibir o resultado
    cv2.imshow("output", OUTPUT)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
