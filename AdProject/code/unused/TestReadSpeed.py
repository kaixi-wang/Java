import datetime
from MyImage import MyImage 
from multiprocessing import Pool, TimeoutError, Process
def processImage(mBytes, buffer):
    ind = 0
    for y in range(0, height):
        for x in range(0, width):
            buffer[ind*3 + 0] = mBytes[ind]
            buffer[ind*3 + 1] = mBytes[ind+height*width]
            buffer[ind*3 + 2] = mBytes[ind+height*width*2]
            ind += 1

initialTime = datetime.datetime.now()

file = open('dataset/data_test1.rgb', 'rb')
height = 270
width = 480

pool = {}
for i in range(0, 100):
    mBytes = file.read(480 * 270 * 3)
    buffer = bytearray([0]) * width * height * 3
    pool[i] = Process(target=processImage, args=(mBytes, buffer))
    pool[i].start()

for i in range(0, 100):
    pool[i].join()

finalTime = datetime.datetime.now()

print(initialTime)
print(finalTime)
print(finalTime - initialTime)