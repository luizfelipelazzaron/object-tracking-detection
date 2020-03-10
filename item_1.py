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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
pi = np.pi
lower = 0
upper = 1
font = cv2.FONT_HERSHEY_SIMPLEX
font2 = cv2.FONT_HERSHEY_PLAIN

print("Press q to QUIT")

def auto_canny(image, sigma=0.33):
    """apply automatic Canny edge detection using the computed median"""
    v = np.median(image)
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

    #hsv's do branco
    hsv1, hsv2 = aux.ranges('#ffffff')
    
    #Mask do branco
    mask = cv2.inRange(frame, hsv1, hsv2)
    
    ##Seleção
    edges = cv2.Canny(gray, 50, 200)
    
    # Detect the edges present in the image

    # A gaussian blur to get rid of the noise in the image
    # blur = cv2.GaussianBlur(gray,(5,5),0)
    
    # bordas = auto_canny(blur)
    
    lines = cv2.HoughLinesP(edges, 1, pi/180, 100, minLineLength=80, maxLineGap=5)
    print(lines.shape)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        try:
            tangente = (y2-y1)/(x2-x1) # inclinação
            if abs(tangente) > 0.6 and abs(tangente) < 0.9:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        except:
            pass

    OUTPUT = frame

    # Draw Lines
    lines = None
    lines = cv2.HoughLines(bordas, 20, pi/180, 10, None, 0, 0)  
    
    if lines is not None:      
        lines = np.uint16(np.around(lines))
        print(lines.shape)

        for i in range(0, len(lines)):

            rho = lines[i][0][0]
            theta = lines[i][0][1]

            a,b = math.cos(theta), math.sin(theta)
            x0,y0 = a*rho, b*rho

            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            # draw the outer line
            cv2.line(OUTPUT, pt1, pt2, (0,255,0), 2, cv2.LINE_AA)

    # Linha default
    # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
    try: 
        # cv2.line(frame, (lines[0], lines[1]), (0,255,0), 2)
        cv2.line(blur, (lines[0], lines[1]), (0,255,0), 2)        
    except:
        pass
    
    # adicionar textos na tela:
    cv2.putText(OUTPUT," Aperte q para sair", (0,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    # Display the resulting frame
    cv2.imshow("outpu", OUTPUT)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
