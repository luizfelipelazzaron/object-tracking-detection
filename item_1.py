#importações do Vídeo
import cv2
import numpy as np
import time

#Importações para a Máscara
import auxiliar as aux
import math

<<<<<<< HEAD
def retas(A, B): # Pontos A e B pertencentes a reta
    # m = By - Ay / Bx - Ax
    c_angular = (B[1] - A[1]) / (B[0] - A[0])
    # h = Ay - m.Ax
    c_linear = A[1] - (c_angular* A[0])

    return [c_angular, c_linear]

def pontos(reta1, reta2):
    X = (((reta1[1] - reta2[1]) / (reta2[0] - reta1[0])))
    Y = ((reta1[0] * X) + reta1[1])

    return [X,Y]

=======
# If you want to open a video, just change v2.VideoCapture(0) from 0 to the filename, just like below
# cap = cv2.VideoCapture('VID_20200302_063445951.mp4')
# cap = cv2.VideoCapture('VID_20200302_063554327.mp4')
cap = cv2.VideoCapture('VID_20200302_063719050.mp4')

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
>>>>>>> 9c2c4391e706de508715645d196ae3debf7e11d6
def auto_canny(image, sigma=0.33):
    """apply automatic Canny edge detection using the computed median"""
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

<<<<<<< HEAD
# leitura do video
cap = cv2.VideoCapture('VID_20200302_063445951.mp4')
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pi = np.pi
lower = 0
upper = 1
font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_PLAIN

# print("Press q to QUIT")
=======
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
        
def tangente(linha) -> float:
    "Devolve a tangente do ângulo formado entre a linha e a horizontal (no referencial da imagem)"
    x1,y1,x2,y2 = linha[0]
    return (y2-y1)/(x2-x1)

def draw_line(img, linha, color:str):
    color = colors[color]
    x1,y1,x2,y2 = linha[0]
    cv2.line(img, (x1, y1), (x2, y2), color, 2) 
>>>>>>> 9c2c4391e706de508715645d196ae3debf7e11d6

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
<<<<<<< HEAD
    frame_hsv = cv2.cvtColor (frame, cv2.COLOR_BGR2HSV)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # hsv's do branco
    hsv1, hsv2 = aux.ranges('#ffffff')
    
    # Mask do branco
    mask = cv2.inRange(frame, hsv1, hsv2)
    
    # Seleção
    edges = cv2.Canny(gray, 50, 200)
    
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)   
    bordas = auto_canny(blur)
   
    # Draw Lines
    # lines = None
    # Usar modelo probabilistico
    lines = cv2.HoughLinesP(bordas, 40, pi/180, 90, None, 35, 5)  
    
    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    c_linear = []
    c_angular = []
    if lines is not None:
        for l in lines:
            i = l[0]
            reta = retas((i[0], i[1]), (i[2],i[3]))  
            c_linear.append(reta[1])
            c_angular.append(reta[0])
            X1.append(i[0])
            Y1.append(i[1])
            X2.append(i[2])
            Y2.append(i[3])
            

    if len(c_angular) >= 2:
        Ang_max = max(c_angular)
        index_max = c_angular.index(Ang_max)
        Lin_max = (c_linear[index_max])
        Reta_max = (Ang_max,Lin_max)
        Ang_min = min(c_angular)
        index_min = c_angular.index(Ang_min)
        Lin_min = (c_linear[index_min])
        Reta_min = (Ang_min,Lin_min)
        Ponto = pontos(Reta_max,Reta_min)
        Pix1 = int(X1[index_max])
        piy1 = int(Y1[index_max])
        Pix2 = int(X1[index_min])
        piy2 = int(Y1[index_min])


        if not math.isnan(Ponto[0]) and not math.isnan(Ponto[1]):
                # cv2.circle(frame,(int(Ponto[0]),int(Ponto[1])),15,(0,255,0),2)
                cv2.line(frame, (Pix1, piy1), (int(Ponto[0]),int(Ponto[1])), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(frame, (Pix2, piy2), (int(Ponto[0]),int(Ponto[1])), (0, 255, 0), 2, cv2.LINE_AA)
                # cv2.circle(frame,(int(Ponto[0]),int(Ponto[1])),2,(0,0,255),3)

    # if lines is not None:      
    #     lines = np.uint16(np.around(lines))
    #     print(lines)

    #     for i in range(0, len(lines)):

    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]

    #         a,b = math.cos(theta), math.sin(theta)
    #         x0,y0 = a*rho, b*rho

    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

    #         # draw the outer line
    #         cv2.line(frame, pt1, pt2, (0,255,0), 2, cv2.LINE_AA)

    # # Linha default
    # # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    # try: 
    #     # cv2.line(frame, (lines[0], lines[1]), (0,255,0), 2)
    #     cv2.line(blur, (lines[0], lines[1]), (0,255,0), 2)        
    # except:
    #     pass
    
    # # adicionar textos na tela:
    # cv2.putText(frame," Aperte q para sair", (0,50), font, 1,(255,255,255),2,cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("Frame", frame)
=======
    
    # Conversão para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Redução de ruído 1
    gray = cv2.GaussianBlur(gray,(5,5), 0.33)

    # limiarização da imagem ( fonte: https://youtu.be/P2R7Nn1_VwQ )
    limiar,img_limiar = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Redução de ruido 2
    # Aplicação de um kernel personalizado ( fonte: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html)
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

>>>>>>> 9c2c4391e706de508715645d196ae3debf7e11d6
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
