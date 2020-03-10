#importações do Vídeo
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

#Importações para a Máscara
from ipywidgets import widgets, interact, IntSlider
import auxiliar as aux
import math

# If you want to open a video, just change v2.VideoCapture(0) from 0 to the filename, just like below
cap = cv2.VideoCapture('VID_20200302_063445951.mp4')

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

lower = 0
upper = 1
font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_PLAIN

print("Press q to QUIT")

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    frame_hsv = cv2.cvtColor (frame, cv2.COLOR_BGR2HSV)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ##hsv's do branco
    hsv1, hsv2 = aux.ranges('#FFFFFF')
    
    ##Mask do branco
    mask = cv2.inRange(frame_hsv, hsv1, hsv2)
    
    ##Seleção
    segmentado = cv2.morphologyEx(mask,cv2.MORPH_RECT, np.ones((5, 5)))
    selecao = cv2.bitwise_and(frame, frame, mask=segmentado)
    
    # A gaussian blur to get rid of the noise in the image
    blur = cv2.GaussianBlur(gray,(5,5),0)
    
    # Detect the edges present in the image
    bordas = auto_canny(blur)

    # Obtains a version of the edges image where we can draw in color
    #cdst
    bordas_color = cv2.cvtColor(bordas, cv2.COLOR_GRAY2BGR)
    # cdstP = np.copy(bordas_color)

    # HoughCircles - detects circles using the Hough Method. For an explanation of
    # param1 and param2 please see an explanation here http://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
    # lines = None
    lines = cv2.HoughLines(bordas,1,np.pi/180, 150, None, 0, 0)  
    # circles = cv2.HoughCircles(bordas,cv2.HOUGH_GRADIENT,2,40,param1=50,param2=100,minRadius=5,maxRadius=60)

    if lines is not None:      
        lines = np.uint16(np.around(lines))
        #print(lines.shape)
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a*rho
            y0 = b*rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            # draw the outer line
            cv2.line(frame, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)
            cv2.line(selecao, pt1, pt2, (0,255,0), 3, cv2.LINE_AA)

    # Draw a diagonal blue line with thickness of 5 px
    # cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.line(bordas_color, (0,0), (511,511), (255,0,0), 5)

    # cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
    cv2.rectangle(bordas_color,(0,0),(250,250),(0,255,0),3)

    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    try: 
        cv2.line(frame, (lines[0], lines[1]), (0,255,0), 2)

        # circle1 = np.array(lines[0], dtype='int64') # converter para int64 antes de realizar operações
        # vetor1_2 = (circle1-circle2)
        # tangente = -vetor1_2[1]/vetor1_2[0]
        # angulo = math.degrees( math.atan(tangente) )
        # f = 617.39 #17.0
        # H = 13.8
        # h = math.sqrt( vetor1_2.dot( vetor1_2 ) ) # forma otimizada de calcular distancia
        # variavel = round( (H * f / h), 2)
        
    except:
        pass
    
    # adicionar textos na tela:

    cv2.putText(frame," Aperte q", (0,50), font, 1,(0,0,0),2,cv2.LINE_AA)
    # cv2.putText(frame,"  distancia = "+str(variavel)+" cm",(0,120), font2, 1.2, (0,0,0), 2, cv2.LINE_AA)
    # cv2.putText(frame,"  angulo = %.0f degrees" %angulo,(0,140), font2, 1.2, (0,0,0), 2, cv2.LINE_AA)
    # cv2.putText(frame,"  h = %.0f px" %h,(0,160), font2, 1.2,(0,0,0), 2, cv2.LINE_AA)
    cv2.putText(selecao," Aperte q para sair", (0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    # cv2.putText(selecao,"  distancia = "+str(variavel)+" cm",(0,120), font2, 1.2, (255,255,255), 2, cv2.LINE_AA)
    # cv2.putText(selecao,"  angulo = %.0f degrees" %angulo,(0,140), font2, 1.2, (255,255,255), 2, cv2.LINE_AA)
    # cv2.putText(selecao,"  h = %.0f px" %h,(0,160), font2, 1.2,(255,255,255), 2, cv2.LINE_AA)

    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    # Display the resulting frame
    # cv2.imshow('Detector de circulos', selecao)
    cv2.imshow("Frame", frame)
    # cv2.imshow("mask", bordas_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
