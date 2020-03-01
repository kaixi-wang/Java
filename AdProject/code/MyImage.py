import cv2 as cv, cv2
import imutils
import numpy as np
from sklearn.cluster import KMeans
from AffineSift import detect_image
from ASizeTemplate import affine_size_template_match


# import matplotlib.pyplot as plt


def read_rgb(filename, height=270, width=480):
    print("Reading videofile...")
    f = open(filename, "r")
    arr = np.fromfile(f, dtype=np.uint8)
    num_frames=int(arr.shape[0]/(height*width*3)) #9000
    arr = arr.reshape(num_frames, (height * width * 3))
    frames=[]
    for frame_number in np.arange(num_frames):
        #r=arr[frame_number][0:(width*height)]
        #g=arr[frame_number][(width*height):(2*width*height)]
        #b=arr[frame_number][(2*width*height):]
        #rgb = np.array(list(zip(r, g, b)))
        #rgb_frame = rgb.reshape((height, width, 3))
        r = arr[frame_number][0:(width * height)].reshape((height, width))
        g = arr[frame_number][(width * height):(2 * width * height)].reshape((height, width))
        b = arr[frame_number][(2 * width * height):].reshape((height, width))
        rgb_frame = np.dstack([r, g, b])
        #rgb_frame= rgb_frame.reshape((num_frames, 270,480,3))
        frames.append(rgb_frame)#.T)
        if frame_number%500==0:
            print('{} percent complete...'.format(frame_number*1.0/num_frames))
    return np.array(frames)

class MyImage:
    width = 480
    height = 270
    mBytes = None

    def __init__(self):
        self.buffer = bytearray([0]) * self.width * self.height * 3
        return
        
    def ReadImage(self, bytes):
        self.mBytes = bytes
        height = self.height
        width = self.width
        ind = 0
        for y in range(0, height):
            for x in range(0, width):
                a = 0
                r = self.mBytes[ind]
                g = self.mBytes[ind+height*width]
                b = self.mBytes[ind+height*width*2]
                self.buffer[ind*3 + 0] = r
                self.buffer[ind*3 + 1] = g
                self.buffer[ind*3 + 2] = b
                ind += 1
        

    def GetBuffer(self):
        return self.buffer
    
    def GetPixel(self, x, y, channel):
        index = self._CoordToIndex(x, y)
        return self.buffer[index + channel]
    
    def _CoordToIndex(self, x, y):
        return (self.width * y + x ) * 3
    
    def _ToNpArray(self):
        img = np.zeros((270, 480, 3), dtype=np.uint8)
        ind = 0
        height = 270
        width = 480
        for y in range(0, 270):
            for x in range(0, 480):
                r = self.mBytes[ind]
                g = self.mBytes[ind+height*width]
                b = self.mBytes[ind+height*width*2]
                img[y][x][0] = r
                img[y][x][1] = g
                img[y][x][2] = b
                ind += 1
        return img
    
    def ASiftMatch(self, idx, logo_path, output_dir):
        video_image = self._ToNpArray()
        video_image = cv.cvtColor(video_image, cv2.COLOR_RGB2GRAY)
        
        logo_image = read_rgb(logo_path)[0]
        logo_image = cv.cvtColor(logo_image, cv2.COLOR_RGB2GRAY)

        vis, corners, pairs = detect_image(logo_image, video_image)
        print("Processing Frame " + str(idx))
        corners_size_appropriate = True
        
        w = 480
        h = 270

        found = False
        # Detect if 1. cornors are way out of bound, 2. corners are not too close
        for i in range (0, len(corners)):
            if corners[i][0] < -w or corners[i][0] > 2*w or corners[i][1] < -h or corners[i][1] > 2*h:
                corners_size_appropriate = False
            for j in range(i+1, len(corners)):
                dist = np.linalg.norm(corners[j] - corners[i])
                if (dist < 10):
                    corners_size_appropriate = False

        if (len(pairs) > 10 and corners_size_appropriate):
            cv.imwrite(output_dir + str(idx) + "_" + str(len(pairs)) + ".jpg" , vis)
            found = True
        
        return found

    def ATemplate(self, idx, logo_path):
        video_image = self._ToNpArray()
        logo_image = read_rgb(logo_path)[0]
        return affine_size_template_match(logo_image, video_image, idx)




