"""
Autor = R&D 
Fecha = Diciembre 12 de 2020
Proyecto =  Conteo 
Empresa = Vida18
"""

import cv2
import math
import numpy as np
import os 
import sys

def split_image(image=None,split=4,position=0):
    '''
    Function divides the orignal image in 4 or 16 smaller ones

    Parameters:
    image -- The original image
    split -- The integer number that divides the image
    position -- The cuadrant of the image

    Exceptions :
    TypeError -- If image==None 
    ValueError -- if split is different to 4 or 16 and position is major than split
    '''     
    try:
        height,width= image.shape[:2]
        if (split == 4 or split == 16):
            divider=int(math.sqrt(split))
            height_divider=int(height/divider)
            width_divider=int(width/divider)
            if position<split :
                dx=int(math.modf(position/divider)[0]*divider*width_divider)
                dy=int(position/divider)*height_divider
                crop_img=image[dy:dy+height_divider,dx:dx+width_divider]
            else:
                raise ValueError ('Position Outside')
            return crop_img,dx,dy
        else:
            raise ValueError ('Split Outside')
    except TypeError as e:
        print(e)
        print('Video or Image empty')
        sys.exit()
    except Exception as e:
        print(e,type(e))
        sys.exit()


def color_class(name_class):

    switcher ={
        "rose": (0,0,255),
        "arveja":(0,255,0),
        "palmiche":(255,0,0),
        "rayando_color":(17,70,244),
        "malla":(255,255,0),
        "conteo":(255,255,255)
    }
    return switcher.get(name_class,(0,0,0))

def draw_box(image,clases,x,y,percentage,counted,total_count):
    try:
        for i in range(len(clases)):
            start_point=(int(x[i][0]),int(y[i][0]))
            end_point=(int(x[i][1]),int(y[i][1]))
            if counted[i] :
                color=color_class('conteo')
            else:
                color = color_class(clases[i])
            if int(x[i][1]) >= 3200:
                xpoint= int(x[i][1]) - 50
            else :
                xpoint= int(x[i][1])
            cv2.rectangle(image,start_point,end_point,color,4)
            cv2.putText(image,f'{clases[i]} {percentage[i]}%' ,(xpoint,int(y[i][0])-10),cv2.FONT_HERSHEY_COMPLEX,1.5,color,4)

        show_c=f'Arvejas: {total_count["arveja"]}, Palmiche: {total_count["palmiche"]},Rayando Color: {total_count["rayando_color"]},Rosa: {total_count["rose"]}, Malla: {total_count["malla"]} '
        cv2.putText(image,show_c, (10, 200), cv2.FONT_HERSHEY_PLAIN,  3, (255,255,0), 4,lineType=cv2.LINE_AA)
        return image
    except Exception as e:
        print('Error drawing boxes')
        print(e, type(e))
        sys.exit()

def video(path_video):
    try:
        cap = cv2.VideoCapture(path_video)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        if h==0:
            raise TypeError ("Video don't exist")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
        fps = cap.get(cv2.CAP_PROP_FPS) 

        return cap,h,w,fps
    except Exception as e:
        print(e)
        sys.exit()
def make_video(name_video,h,w,fps):
    try:
       
        fourcc = cv2.VideoWriter_fourcc(*'XVID') #Video Format (avi)
        
        out = cv2.VideoWriter(f'./detections/{name_video}.avi',fourcc, fps, (w ,h))  #Write new

        return out
    except Exception as e:
        print('Error Writing Video')
        print(e)
        sys.exit()
