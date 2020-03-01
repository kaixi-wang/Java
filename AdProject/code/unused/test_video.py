# import cv2
import sys
import scipy
import chardet
from array import *
from PIL import Image
import numpy as np
import os

import matplotlib
import matplotlib.pyplot as plt
import datetime

import matplotlib.gridspec as gridspec
import matplotlib.pylab as pl

# Date
now = datetime.datetime.now()

# Scene changes:
#       - frame # 2400
#       - frame # 2850

height = 270
width = 480
frameLength = height * width * 3

array_r = [[0 for x in range(width)] for y in range(height)] 
array_g = [[0 for x in range(width)] for y in range(height)] 
array_b = [[0 for x in range(width)] for y in range(height)] 
array_Y = [[0 for x in range(width)] for y in range(height)] 
array_L = [[0 for x in range(width)] for y in range(height)] 

frameL_array = [0]
meanR_array = []
meanG_array = []
meanB_array = []

frameYdifferences = [0] * width*height
frameLdifferences = [0] * width*height

frame_intensity = 0
rgbArray = np.zeros((height,width,3), 'uint8')
gamma = 2.2

outlog_name = now.strftime("%d-%B-%Y_%H-%M-%S") + '.txt'
# outlog_name = "OUT.txt"
log_file = open(outlog_name, "w")

r_file_name = "r_log.txt"
g_file_name = "g_log.txt"
b_file_name = "b_log.txt"
l_file_name = "l_log.txt"
r_file = open(r_file_name, "w")
g_file = open(g_file_name, "w")
b_file = open(b_file_name, "w")
l_file = open(l_file_name, "w")


cmp_mean_array = []
deltaF = 1



def saveFrameAsImage(fileName):
    img = Image.fromarray(rgbArray)
    img.save(fileName)

def processFrame(start, n):
    global rgbArray, log_file, r_file, g_file, b_file

    with open("data_test1.rgb", "rb") as f:

        i = 0
        cmp_frame_avg_sum = 0

        while i < deltaF:

            f.seek(start + height*width*3*i)
            bytes_img = f.read(frameLength)
            sumY = 0
            sumLightness = 0
            sumR = 0
            sumG = 0
            sumB = 0
            
        
            for y in range(height):
                for x in range(width):
                
                    ind = y * width + x
                    a = 0
                    r = bytes_img[ind]
                    g = bytes_img[ind+height*width]
                    b = bytes_img[ind+height*width*2]
                    # print("------------------------------------------")
                    # print("(", x , ", ", y, ")")
                    # if (x==0 and y==0):
                    # print("r: ", r)
                    # print("g: ", g)
                    # print("b: ", b)

                    array_r[y][x] = r
                    array_g[y][x] = g
                    array_b[y][x] = b
                    # Get luminance to use as average intensity value of each frame.
                    Y = .2126 * r**gamma + .7152 * g**gamma + .0722 * b**gamma
                    L = 116 * (Y**(1/3)) - 16

                    sumY += Y
                    sumLightness += L
                    sumR += r
                    sumG += g
                    sumB += b

                    array_Y[y][x] = Y
                    array_L[y][x] = L
                    

            rgbArray[..., 0] = array_r
            rgbArray[..., 1] = array_g
            rgbArray[..., 2] = array_b
            
            # Find avg Y of this frame.
            meanY = sumY / (width*height)
            # frameYdifferences[n] = meanY 


            avgR = sumR / (width*height)
            avgG = sumG / (width*height)
            avgB = sumB / (width*height)

            r_file.write(str(n) + ", ")
            g_file.write(str(n) + ", ")
            b_file.write(str(n) + ", ")

            r_file.write(str(avgR) + "\n")
            g_file.write(str(avgG) + "\n")
            b_file.write(str(avgB) + "\n")
            

            meanR_array.append(avgR)
            meanG_array.append(avgG)
            meanB_array.append(avgB)
            
            # frameLdifferences[n] = meanL
            meanL = sumLightness / (width*height)
            l_file.write(str(n) + ", ")
            l_file.write(str(meanL) + "\n")

            frameL_array.append(meanL)

            cmp_frame_avg_sum += meanL

            diffL = meanL - frameL_array[len(frameL_array) - 2]
            print("meanL: ", meanL)

            # imgOutName = "myImg" + str(n) + ".jpeg"
            # saveFrameAsImage(imgOutName)

            diffR = avgR - meanR_array[len(meanR_array) - 2]

            if abs(diffR) > 9:
                imgOutName = "myImg" + str(n) + ".jpeg"
                saveFrameAsImage(imgOutName)
                print("***")
                log_file.write("------> ")
            
            line = str(n) + ": " + str(meanL)
            line2 = "     - diff: " + str(abs(diffL))
            print(line)
            print(line2)
            print("")
            log_file.write(line)
            log_file.write("\n")
            log_file.write(line2)
            log_file.write("\n")
            log_file.write("\n")

            i = i + 1

        cmp_avg = cmp_frame_avg_sum / 5
        cmp_mean_array.append(cmp_avg)

        # if n > 2390 and abs(meanY - frameYdifferences[n-1]) > 10000:
        #     imgOutName = "myImg" + str(n) + ".jpeg"
        #     saveFrameAsImage(imgOutName)
        #     print("***")
        #     log_file.write("------> ")
        
        # line = str(n) + ": " + str(meanY)
        # line2 = "     - diff: " + str(abs(meanY-frameYdifferences[n-1]))
        # print(line)
        # print(line2)
        # print("")
        # log_file.write(line)
        # log_file.write("\n")
        # log_file.write(line2)
        # log_file.write("\n")
        # log_file.write("\n")
            

                
    # Compare average intensity frame by frame for significant changes which signal scene changes.

    # 2 methods for Scene detection
    # - content-based detection
    # - threshold-based detection
    # Content-based detection seems for applicable to our case.



def processVideo():
    # Detect advertisements to remove.
    # f = open("data_test1.rgb", "rb")
    numFrames = int(os.path.getsize("data_test1.rgb") / (height*width*3))
    print(numFrames)

    n = 2390
    start = n * height*width*3
    processFrame(start,n)

    startFrame = 1
    endFrame = 9000

    

    for n in range(startFrame, endFrame, deltaF):
        start = n * height*width*3
        # print(n)
        # print(start)
        processFrame(start, n)


        # if n == 2750:
        #     break
    

    # bytes_img = f.read(len)

    # for i in range(1, numFrames):
    #     print(i, ": ", frameYdifferences[i] - frameYdifferences[i-1])
    #     if i == 2750:
    #         break

    print('*DONE*')
    # print(frameL_array)
    # xAxis = list(range(startFrame-2, endFrame))
    # plt.plot(xAxis, frameL_array)
    # plt.show()

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 2)
    pl.figure()
    


    # Plot R
    # plt.subplot(3, 1, 1)
    xAxis = list(range(startFrame-1, endFrame, deltaF))
    ax = pl.subplot(gs[0, 0]) # row 0, col 0
    pl.plot(xAxis, meanR_array)
    # plt.plot(xAxis, meanR_array)

    # Plot G
    # plt.subplot(3, 1, 2)
    # xAxis = list(range(startFrame-2, endFrame, deltaF))
    # plt.plot()
    xAxis = list(range(startFrame-1, endFrame, deltaF))
    ax = pl.subplot(gs[0, 1]) # row 0, col 0
    pl.plot(xAxis, meanG_array)


    # Plot B
    xAxis = list(range(startFrame-1, endFrame, deltaF))
    ax = pl.subplot(gs[1, 1]) # row 0, col 0
    pl.plot(xAxis, meanB_array)


    plt.show()


    # val = 0. # this is the value where you want the data to appear on the y-axis.
    # ar = np.arange(10) # just as an example array
    # plt.plot(ar, np.zeros_like(ar) + val, 'x')
    # plt.show()



            

# 2. Read frames



# We might need a buffer for video storage.

def main():
    processVideo()
    # processFrame()
    # for y in range(height):
    #     for x in range(width):
    #         print("Y: ", array_Y[y][x])
    r_file.close()
    g_file.close()
    b_file.close()

if __name__ == '__main__':
    main()