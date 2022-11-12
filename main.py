# This is a sample Python script.
import gzip
import pickle

import numba
from numba import jit, cuda

import numpy as np
from math import *
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
@jit(nopython=True)
def relu(x):
    return max(0.0,x)
@jit(nopython=True)
def relu_deriv(x):
    if x > 0:
        x = 1
    elif x <= 0:
        x = 0
    return x
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
@jit(target_backend='cuda')
def prosta_siec(input,expected_output,learning_rate,weight,output_weight):
    image_sections = []
    kernal_size = sqrt(len(weight[0]))
    for i in range(input.shape[0] - kernal_size + 1):
        for j in range((input.shape[1]-kernal_size+1)):
            tmp = []
            for k in range(kernal_size):
                tmp.append(input[i:i + kernal_size, j + k])
            image_sections.append([element for row in tmp for element in row])

    image_sections = np.array(image_sections)

    kernel_layer = np.dot(image_sections, weight.T)
    kernel_layer_flatten = kernel_layer.flatten()
    kernel_layer_flatten = kernel_layer_flatten.reshape(len(kernel_layer_flatten), 1)
    kernel_layer_flatten_tmp = kernel_layer_flatten.copy()
    # kernel_layer_flatten = np.vectorize(relu)(kernel_layer_flatten)
    kernel_layer_flatten=np.maximum(0,kernel_layer_flatten)
    layer_output = np.dot(output_weight, kernel_layer_flatten)
    layer_output = layer_output.reshape(len(layer_output), 1)
    layer_output_delta = 2 * (layer_output - expected_output) / len(layer_output)
    kernel_layer_delta = np.dot(output_weight.T, layer_output_delta)
    # kernel_layer_delta*= np.vectorize(relu_deriv)(kernel_layer_flatten_tmp)
    kernel_layer_delta*=np.where(kernel_layer_flatten_tmp>0,1,0)



    kernel_layer_delta_reshaped = kernel_layer_delta.reshape(kernel_layer.shape)

    layer_output_weight_delta = np.dot(layer_output_delta, kernel_layer_flatten.T)

    kernel_layer_weight_delta = np.dot(kernel_layer_delta_reshaped.T, image_sections)

    output_weight = output_weight - learning_rate * layer_output_weight_delta
    weight = weight - learning_rate * kernel_layer_weight_delta

    return layer_output,output_weight,weight






def przyklad1():
    dane_wejsciowe=np.array([[8.5,0.65,1.2],[9.5,0.8,1.3],[9.9,0.8,0.5],[9.0,0.9,1.0]])
    oczekiwane_wyjscie=np.array([[0],[1]])
    kernel1_weight=np.array([0.1,0.2,-0.1,-0.1,0.1,0.9,0.1,0.4,0.1])
    kernel2_weight=np.array([0.3,1.1,-0.3,0.1,0.2,0.0,0.0,1.3,0.1])
    output_weight=np.array([[0.1,-0.2,0.1,0.3],[0.2,0.1,0.5,-0.3]])
    weight=np.array([kernel1_weight,kernel2_weight])
    prosta_siec(dane_wejsciowe, oczekiwane_wyjscie, 0.01,weight , output_weight)

    image_sections=[]
    kernal_size = isqrt(len(kernel1_weight))
    for i in range(ceil(dane_wejsciowe.shape[0]/kernal_size)):
        for j in range(ceil(dane_wejsciowe.shape[1]/kernal_size)):
            tmp=[]
            for k in range(kernal_size):
                tmp.append(dane_wejsciowe[i:i+kernal_size,j+k])
            image_sections.append([element for row in tmp for element in row])


    image_sections=np.array(image_sections)
    kernels=np.array([kernel1_weight,kernel2_weight])
    kernel_layer=np.dot(image_sections,kernels.T)
    print(kernel_layer)
    kernel_layer_flatten=kernel_layer.flatten()
    print(kernel_layer_flatten)
    kernel_layer_flatten=kernel_layer_flatten.reshape(len(kernel_layer_flatten),1)
    print(kernel_layer)
    layer_output=np.dot(output_weight,kernel_layer_flatten)
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
@jit(target_backend='cuda')
def predict(input,weight,output_weight):
    image_sections = []
    kernal_size = sqrt(len(weight[0]))
    for i in range(input.shape[0] - kernal_size + 1):
        for j in range((input.shape[1]-kernal_size+1)):
            tmp = []
            for k in range(kernal_size):
                tmp.append(input[i:i + kernal_size, j + k])
            image_sections.append([element for row in tmp for element in row])

    image_sections = np.array(image_sections)

    kernel_layer = np.dot(image_sections, weight.T)
    kernel_layer_flatten = kernel_layer.flatten()
    kernel_layer_flatten = kernel_layer_flatten.reshape(len(kernel_layer_flatten), 1)
    kernel_layer_flatten_tmp = kernel_layer_flatten.copy()
    # kernel_layer_flatten = np.vectorize(relu)(kernel_layer_flatten)
    kernel_layer_flatten=np.maximum(0,kernel_layer_flatten)
    layer_output = np.dot(output_weight, kernel_layer_flatten)
    layer_output = layer_output.reshape(len(layer_output), 1)
    return layer_output



def zadanie1():
    obraz_wejsciowy = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    filtr = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    krok = 1
    padding = 0
    obraz_wyjsciowy = splot(obraz_wejsciowy, filtr, krok, True)
    print(obraz_wyjsciowy)





if __name__ == '__main__':
    przyklad1()
    TRAINING_SIZE = 1000
    PREDICT_COUNT = 10000
    EPOCHS = 1000
    BATCH = 100
    ALPHA = 0.2

    f = open("train-images.idx3-ubyte", 'rb')
    labe = open("train-labels.idx1-ubyte", 'rb')
    f.seek(16)
    labe.seek(8)
    list_image = np.zeros((TRAINING_SIZE, 28, 28))
    list_label = np.zeros((TRAINING_SIZE, 10,1))
    for i in range(TRAINING_SIZE):
        for j in range(28):
            for k in range(28):
                list_image[i][j][k] = int.from_bytes(f.read(1), byteorder='big')/255
        list_label[i][int.from_bytes(labe.read(1), byteorder='big')] = 1
    f.close()
    labe.close()
    f = open("t10k-images.idx3-ubyte", 'rb')
    labe = open("t10k-labels.idx1-ubyte", 'rb')
    f.seek(16)
    labe.seek(8)
    list_image_test = np.zeros((PREDICT_COUNT, 28, 28))
    list_label_test = np.zeros((PREDICT_COUNT, 10, 1))
    for i in range(PREDICT_COUNT):
        for j in range(28):
            for k in range(28):
                list_image_test[i][j][k] = int.from_bytes(f.read(1), byteorder='big') / 255
        list_label_test[i][int.from_bytes(labe.read(1), byteorder='big')] = 1
    f.close()
    labe.close()



    image_size_x = 28
    image_size_y = 28
    wagi = np.random.uniform(-0.01,0.01,(16,9))
    waga_wyjsciowa = np.random.uniform(-0.1,0.1,(10,(image_size_x-2)*(image_size_y-2)*16))
    for i in range(EPOCHS):
        sum = 0
        n = 0
        for j in range(TRAINING_SIZE):

            kernel,waga_wyjsciowa,wagi=prosta_siec(list_image[j],list_label[j],0.01,wagi,waga_wyjsciowa)
            if np.argmax(kernel)==np.argmax(list_label[j]):
                sum+=1
            n+=1

        print("epoch: ",i," accuracy: ",sum/n*100,"%")

        sum_t = 0
        n_t = 0
        for j in range(PREDICT_COUNT):
            kernel = predict(list_image_test[j], wagi, waga_wyjsciowa)
            if np.argmax(kernel) == np.argmax(list_label_test[j]):
                sum_t += 1
            n_t += 1
        print("accuracy_test: ", sum_t / n_t * 100, "%")











