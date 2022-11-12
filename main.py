# This is a sample Python script.
import numpy as np
from math import *
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def splot(obraz_wejsciowy,filtr,krok,padding):

    obraz_wyjsciowy = np.zeros((obraz_wejsciowy.shape[0],obraz_wejsciowy.shape[1])) if padding== True else\
        np.zeros(((obraz_wejsciowy.shape[0]-filtr.shape[0])//krok+1,(obraz_wejsciowy.shape[1]-filtr.shape[1])//krok+1))
    if padding:
        for i in range(0, obraz_wejsciowy.shape[0] - filtr.shape[0] + 1, krok):
            for j in range(0, obraz_wejsciowy.shape[1] - filtr.shape[1] + 1, krok):
                obraz_wyjsciowy[i, j] = np.sum(obraz_wejsciowy[i:i + filtr.shape[0], j:j + filtr.shape[1]] * filtr)
    else:
        for i in range(obraz_wyjsciowy.shape[0]):
            for j in range(obraz_wyjsciowy.shape[1]):
                obraz_wyjsciowy[i, j] = np.sum(obraz_wejsciowy[i:i + filtr.shape[0], j:j + filtr.shape[1]] * filtr)

    return obraz_wyjsciowy

def siec_konwolucyjna(input,expected_output,learning_rate,weight):
    image_section = 0

def przyklad1():
    dane_wejsciowe=np.array([[8.5,0.65,1.2],[9.5,0.8,1.3],[9.9,0.8,0.5],[9.0,0.9,1.0]])
    oczekiwane_wyjscie=np.array([[0],[1]])
    kernel1_weight=np.array([0.1,0.2,-0.1,-0.1,0.1,0.9,0.1,0.4,0.1])
    kernel2_weight=np.array([0.3,1.1,-0.3,0.1,0.2,0.0,0.0,1.3,0.1])
    output_weight=np.array([[0.1,-0.2,0.1,0.3],[0.2,0.1,0.5,-0.3]])

    image_sections=[]

    for i in range(ceil(dane_wejsciowe.shape[0]/isqrt(len(kernel1_weight)))):
        for j in range(ceil(dane_wejsciowe.shape[1]/isqrt(len(kernel1_weight)))):
            tmp=[]
            for k in range(isqrt(len(kernel1_weight))):
                tmp.append(dane_wejsciowe[i:i+isqrt(len(kernel1_weight)),j*isqrt(len(kernel1_weight))+k])
            image_sections.append([element for row in tmp for element in row])


    image_sections=np.array(image_sections)
    kernels=np.array([kernel1_weight,kernel2_weight])
    kernel_layer=np.dot(image_sections,kernels.T)
    print(kernel_layer)
    kernel_layer_flatten=kernel_layer.flatten()
    print(kernel_layer_flatten)
    kernel_layer_flatten=kernel_layer_flatten.reshape(len(kernel_layer_flatten),1)
    print(kernel_layer)
    layer_output=np.dot(output_weight,kernel_layer_flatten.flatten())
    layer_output=layer_output.reshape(len(layer_output),1)
    print(layer_output)
    layer_output_delta=2*(layer_output-oczekiwane_wyjscie)/len(layer_output)
    print(layer_output_delta)
    kernel_layer_delta=np.dot(output_weight.T,layer_output_delta)


    print(kernel_layer_delta)

    kernel_layer_delta_reshaped=kernel_layer_delta.reshape(kernel_layer.shape)
    print(kernel_layer_delta_reshaped)


    layer_output_weight_delta=np.dot(layer_output_delta,kernel_layer_flatten.T)
    print(layer_output_weight_delta)


    kernel_layer_weight_delta=np.dot(kernel_layer_delta_reshaped.T,image_sections)
    print(kernel_layer_weight_delta)

    output_weight=output_weight-0.01*layer_output_weight_delta
    print(output_weight)
    kernels=kernels-0.01*kernel_layer_weight_delta
    print(kernels)

def zadanie1():
    obraz_wejsciowy = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    filtr = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    krok = 1
    padding = 0
    obraz_wyjsciowy = splot(obraz_wejsciowy, filtr, krok, True)
    print(obraz_wyjsciowy)


if __name__ == '__main__':
    zadanie1()
    przyklad1()




