"""
Autor = R&D 
Fecha = Diciembre 12 de 2020
Proyecto =  Conteo 
Empresa = Vida18
"""

import cv2
import pandas as pd 
import os 
import sys
from absl import app, flags
from absl.flags import FLAGS
from tools import image_tool as it
from tools import detection
from datetime import datetime
import time
import ntpath

# Definición de banderas
flags.DEFINE_string('weights', './weights/yolov4_BEST.weights', 'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to img')
flags.DEFINE_integer('split', 4, 'split image in ')
flags.DEFINE_string('i', './video/804.avi', 'path to images folder')
flags.DEFINE_float('iou', 0.2, 'iou threshold')
flags.DEFINE_float('score', 0.3, 'score threshold')
flags.DEFINE_string('Engine','CPU','CPU ó GPU')
flags.DEFINE_string('labels','./classes/obj.names','path to class_files')
flags.DEFINE_string('config','./Config/yolov4-FENO.cfg','path to config_file')

def main(_argv):
    try:
        os.makedirs('./detections', exist_ok=True)
        name_video=ntpath.basename(FLAGS.i)


        conteo_total={'rose':0,'palmiche':0,'rayando_color':0,'arveja':0,'malla':0 }
        conteo_total=pd.Series(conteo_total)

        Threshold=FLAGS.score  
        nms=FLAGS.iou
        whT=FLAGS.size

        
        # Inicializamos las clases
        class_names=detection.clases_names(FLAGS.labels)

        # Carga de la red neuronal desde Darknet
        # Archivos de configuración del modelo 
        model_file =FLAGS.config
        model_weights = FLAGS.weights
        net = cv2.dnn.readNetFromDarknet(model_file,model_weights)
        
        # Cargamos nuestras capas 
        layerNames = net.getLayerNames()
        # Nombres de nuestras capas de salida
        layerNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]

        if FLAGS.Engine=='CPU':
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


        # Read Video
        video,h,w,fps =it.video(FLAGS.i) #Read Video with OpenCV


        name_output=name_video.split('.')[0]
        print(f'Proccesing {name_video}')
        video_detection=it.make_video(name_output,h,w,fps)


        clases_aux=[]
        x_aux=[]
        y_aux=[]
        inicio=datetime.now()
        frame=0
        clean=False
        while True:

            ret,img=video.read()  #Read Frame

            if not ret:
                break
            if frame % 4 ==0:
                clean=True
            else:
                clean=False
            dx=0
            dy=0
            clases=[]
            coor=[]
            percentage=[]
            position=0
            split=FLAGS.split
            while position<split:
                crop_img,dx,dy=it.split_image(img,split,position)
                blob = cv2.dnn.blobFromImage(crop_img,1/255,(whT,whT),swapRB=True,crop=False)
                net.setInput(blob)


                outputs = net.forward(layerNames)
                
                result=detection.findobjects(outputs,crop_img,class_names,Threshold,nms,dx,dy)
                if len(result[0])>0:
                    clases.append(result[0]) # Objetos
                    coor.append(result[1])
                    percentage.append(result[2])

                position+=1

            clases,x,y,percentage,counted=detection.boundary_condition(clases,coor,percentage)

            if clean:
                conteo_total,counted,clases_aux,x_aux,y_aux=detection.conteo(conteo_total,clases,x,y,clases_aux,x_aux,y_aux,counted,int(w*0.6))
            else :
                conteo_total,counted=detection.conteo(conteo_total,clases,x,y,clases_aux,x_aux,y_aux,counted,int(w*0.6))[:2]
                for i in range(len(clases)):
                    clases_aux.append(clases[i])
                    x_aux.append(x[i])
                    y_aux.append(y[i])
            img=it.draw_box(img,clases,x,y,percentage,counted,conteo_total)

            video_detection.write(img)
            frame+=1
        print(conteo_total)
        final=datetime.now()
        duracion=final-inicio
        video.release()
        video_detection.release()

        t=round(duracion.total_seconds())
        
        if t <= 60:
            print('Tiempo total : ' + str(t) + ' segundos')
        else:
            minutes =  int(t // 60)
            t%= 60
            if len(str(t))==1:
                t='0'+str(t)
            print('Tiempo Total:' + str(minutes) +':'+str(t)+ ' minutos')
    except TypeError as e:
        print(e)
        sys.exit()
    except Exception as e:
        print('Error Main')
        print(e,type(e))
        sys.exit()



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass