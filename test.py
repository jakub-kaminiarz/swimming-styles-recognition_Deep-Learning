# author: @Kaminiarz_Jakub
# SWIMMING-STYLES-RECOGNITION
# test.py

import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import operator

dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1]
filename = dir_path +'/' +image_path
image_size=128
num_channels=3
images = []
# odczyt zdjęcia za pomocą OpenCV
image = cv2.imread(filename)
# Zmiana rozmiarów zdjęcia do pożądanego przez nas rozmiaru
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)
#Wynik sieci ma rozmiar [None image_size image_size num_channels]
x_batch = images.reshape(1, image_size,image_size,num_channels)

# Użyjmy zapisany wczesniej model
sess = tf.Session()
# krok1: utworzmy ponownie graf sieciowy
saver = tf.train.import_meta_graph('dogs-cats-model.meta')
# krok2: zaladujmy wagi zapisane wczesniej w funkcji restore
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph() 33
# Zmienna y_pred to tensor, który jest przewidywaniem utworzonej sieci
y_pred = graph.get_tensor_by_name("y_pred:0")

x= graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, 4))


# tworzenie zmiennej feed_dict, która jest wymagana w celu obliczenia prawdopodobienstwa y_pred

feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)

#wyswietlanie wyniku rozpoznanego przez siec
print("")
print(result)
print(" [zaba] [kraul] [delfin] [grzbiet]")
print("")

if result[0,0]>result[0,1] and result[0,0]>result[0,2] and result[0,0]>result[0,3]:
    print("rozpoznano styl klasyczny z prawdopodobieńśtwem",result[0,0]*100, "%")
elif result[0,1]>result[0,0] and result[0,1]>result[0,2] and result[0,1]>result[0,3]:
    print("rozpoznano styl kraulowy z prawdopodobieńśtwem", result[0,1]*100, "%")
elif result[0,2]>result[0,0] and result[0,2]>result[0,1] and result[0,2]>result[0,3]:
    print("rozpoznano styl delfinowy z prawdopodobieńśtwem", result[0,2]*100, "%")
else: print("rozpoznano styl grzbietowy z prawdopodobieńśtwem", result[0,3]*100, "%")