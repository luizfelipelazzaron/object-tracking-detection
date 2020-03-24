# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

def auto_canny(image, sigma=0.33):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged

def Calc_reta (P1, P2):
    Coef_ang = ((P2[1] - P1[1]) / (P2[0] - P1[0]))
    Coef_lin = (P1[1] - (Coef_ang * P1[0]))
    return [Coef_ang, Coef_lin]

def Encontra_ponto (reta1, reta2):
    X = (((reta1[1] - reta2[1]) / (reta2[0] - reta1[0])))
    Y = ((reta1[0] * X) + reta1[1])
    return [X,Y]

cap = cv2.VideoCapture('VID_20200302_063445951.mp4')
#cap = cv2.VideoCapture('VID_20200302_063554327.mp4')
#cap = cv2.VideoCapture('VID_20200302_063719050.mp4')
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


hsv1_a = np.array([23,65,65], dtype=np.uint8)
hsv2_a = np.array([30, 255, 255], dtype=np.uint8)

while(True):

    ret, frame = cap.read()
    with_frame = frame.shape[0]
    Height_frame = frame.shape[1]
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    maskb = cv2.inRange(img_gray, 210, 255)
    maska = cv2.inRange(img_hsv, hsv1_a, hsv2_a)
    maskp = maskb & maska
    maskf = maskb - maskp
    not_mask = cv2.bitwise_not(maskf)
    res = cv2.bitwise_and(img_gray,img_gray, mask = not_mask)
    res2 = cv2.bitwise_and(frame,frame, mask= maskf)
    img_mask = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    final = cv2.bitwise_or(res2, img_mask)

    blur = cv2.GaussianBlur(maskf,(5,5),0)
    bordas = auto_canny(blur) 


    #lines_list = []
    lines_list = cv2.HoughLinesP(bordas, 50, math.pi/180.0, 100, np.array([]), 35, 5)

    X1 = []
    Y1 = []
    X2 = []
    Y2 = []
    Coef_lin = []
    Coef_ang = []
    if lines_list is not None:
        for l in lines_list:
            i = l[0]
            reta = Calc_reta((i[0], i[1]), (i[2],i[3]))  
            Coef_lin.append(reta[1])
            Coef_ang.append(reta[0])
            X1.append(i[0])
            Y1.append(i[1])
            X2.append(i[2])
            Y2.append(i[3])
            

    if len(Coef_ang) >= 2:
        Ang_max = max(Coef_ang)
        index_max = Coef_ang.index(Ang_max)
        Lin_max = (Coef_lin[index_max])
        Reta_max = (Ang_max,Lin_max)
        Ang_min = min(Coef_ang)
        index_min = Coef_ang.index(Ang_min)
        Lin_min = (Coef_lin[index_min])
        Reta_min = (Ang_min,Lin_min)
        Ponto = Encontra_ponto(Reta_max,Reta_min)
        Pix1 = int(X1[index_max])
        piy1 = int(Y1[index_max])
        Pix2 = int(X1[index_min])
        piy2 = int(Y1[index_min])


        if not math.isnan(Ponto[0]) and not math.isnan(Ponto[1]):
                cv2.circle(final,(int(Ponto[0]),int(Ponto[1])),15,(0,255,0),2)
                cv2.line(final, (Pix1, piy1), (int(Ponto[0]),int(Ponto[1])), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.line(final, (Pix2, piy2), (int(Ponto[0]),int(Ponto[1])), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(final,(int(Ponto[0]),int(Ponto[1])),2,(0,0,255),3)

    

    # Display the resulting frame
    cv2.imshow('frame',final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()