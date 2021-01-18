"""
Autor = R&D 
Fecha = Diciembre 12 de 2020
Proyecto =  Conteo 
Empresa = Vida18
"""

import cv2
import numpy as np 
import pandas as pd
import os 
import sys
  


def clases_names(labels_path):
    try:
        class_names=[]
        with open(labels_path,'rt') as f :
            class_names = f.read().rstrip('\n').split('\n')
        return class_names
    except Exception as e:
        print('Error in labels file')
        print(e,type(e))
        sys.exit()

def boundary_condition(clases,coordinates,percentage):
    try:
        obj,x,y,p,c=link_coordinates(clases,coordinates,percentage)
        pos=0
        if len(obj)>1:
            while pos < len(obj):
                count=pos+1
                while count < len(obj):
                    dy=2*abs(int(y[pos][0])-int(y[pos][1]))
                    dx=2*abs(int(x[pos][0])-int(x[pos][1]))
                    if abs(int(y[pos][0])-int(y[count][0])) < dy and abs(int(y[pos][1])-int(y[count][1]))<dy :
                        if abs(int(x[pos][0])-int(x[count][0])) < dx and abs(int(x[pos][1])-int(x[count][1]))<dx :
                            if p[pos] >= p[count]:
                                del obj[count]
                                del x[count]
                                del y[count]
                                del p[count]
                                del c[count]  
                            else :
                                del obj[pos]
                                del x[pos]
                                del y[pos]
                                del p[pos]
                                del c[pos] 
                    count+=1
                pos+=1
        return obj,x,y,p,c
    except Exception as e:
        print('Error in boundary condition')
        print(e,type(e))
        sys.exit()



def link_coordinates(clases,coordinates,percentage):
    try:
        objetos=[]
        x=[]
        y=[]
        p=[]
        counted=[]
        pos=0
        while pos < len(clases):
            count=0
            while count< len(clases[pos]):
                objetos.append(clases[pos][count])
                x.append(coordinates[pos][count][:2])
                y.append(coordinates[pos][count][2:4])
                p.append(percentage[pos][count])
                counted.append(False)
                count+=1
            pos+=1
        return objetos,x,y,p,counted
    except Exception as e:
        print('Error link coordiantes')
        print(e,type(e))
        sys.exit()





def filter(class_name,confs,coordenada,objetos,percentage,coord):
    try:
        if (class_name=='arveja') and (confs >= 60 ) :#51

            objetos.append(class_name)
            coord.append(coordenada)
            percentage.append(confs)
            
            
        elif (class_name=='rayando_color') and (confs>= 60 ) : #45

            objetos.append(class_name)
            coord.append(coordenada)
            percentage.append(confs)

        elif (class_name=='rose') and (confs >= 80 ) : #67

            objetos.append(class_name)
            coord.append(coordenada)
            percentage.append(confs)


        elif (class_name=='palmiche')and (confs >= 70 ) :

            objetos.append(class_name)
            coord.append(coordenada)
            percentage.append(confs)


        elif class_name=='malla' and (confs >= 50 ):

            objetos.append(class_name)
            coord.append(coordenada)
            percentage.append(confs)

        return objetos,coord,percentage
    except Exception as e:
        print('Error in filter detections')
        print(e,type(e))
        sys.exit()


def findobjects(outputs,img,class_names,Threshold,nms,dx=0,dy=0):
    try:
        hT,wT=img.shape[:2]
        bbox= []
        classIds= []
        confs = []
        for output in outputs:
            for detection in output: 
                scores = detection[5:] # Primeros 5 elementos 
                classId = np.argmax(scores)
                confidence =scores[classId]

                if confidence > Threshold: 
                    w,h = int(detection[2]*wT),int(detection[3]*hT)
                    x,y = int(detection[0]*wT-w/2),int(detection[1]*hT-h/2)
                    bbox.append([x,y,w,h])
                    classIds.append(classId)
                    confs.append(float(confidence))
        
        
        # Supreción de bbox
        indices = cv2.dnn.NMSBoxes(bbox,confs,Threshold,nms)

        percentage=[]
        objetos=[]
        coord=[]
        for i in indices :
            i = i[0]
            box = bbox[i]
            x,y,w,h =box[0],box[1],box[2],box[3]

        
            x0=x+dx
            x1=x+w+dx
            y0=y+dy
            y1=y+h+dy
            coordenada=[x0,x1,y0,y1]

            objetos,coord,percentage=filter(class_names[classIds[i]],int(confs[i]*100),coordenada,objetos,percentage,coord)
            
        return objetos,coord,percentage
    except Exception as e:
        print('Error Finding objects')
        print(e,type(e))
        sys.exit()


def conteo(conteo, clases,x,y,clases_aux,x_aux,y_aux,counted,dx):
    if len(clases_aux)>0 and len(clases)>0:
        for i in range(len(clases)):
            distx=False
            disty=False
            same=False
            if len(clases_aux)>0:
                for j in range(len(clases_aux)):
                    if clases[i]==clases_aux[j]:
                        same=True
                        if abs(int(y[i][0])-int(y_aux[j][0])) > 2*abs(int(y_aux[j][0])-int(y_aux[j][1])) and abs(int(y[i][1])-int(y_aux[j][1]))> 2*abs(int(y_aux[j][0])-int(y_aux[j][1])) :
                            disty=True
                        else:
                            disty=False
                        if abs(int(x[i][0])-int(x_aux[j][0])) > dx  and abs(int(x[i][1])-int(x_aux[j][1])) > dx :
                            distx=True
                        else:
                            distx=False
                    if not distx and not disty and same:
                        del clases_aux[j]
                        del x_aux[j]
                        del y_aux[j]
                        break
                if same:
                    if (disty or distx):
                        conteo[clases[i]]=conteo[clases[i]]+1
                        counted[i]=True
                else:
                    conteo[clases[i]]=conteo[clases[i]]+1
                    counted[i]=True
            else :
                conteo[clases[i]]=conteo[clases[i]]+1
                counted[i]=True
                
    elif len(clases)>0 and  len(clases_aux)==0:

        for i in range(len(clases)):
            conteo[clases[i]]=conteo[clases[i]]+1
            counted[i]=True
    else:
        return conteo,counted,clases,x,y
    return conteo,counted,clases,x,y