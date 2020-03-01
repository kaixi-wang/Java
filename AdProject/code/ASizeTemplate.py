import cv2 as cv, cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
import sys
from AffineSift import detect_image
from CV_SIFT_matching import sift_match
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
'''def read_rgb(filename, height=270, width=480):
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

        r = arr[frame_number][0:(width * height)]#.reshape((height, width))
        g = arr[frame_number][(width * height):(2 * width * height)]#.reshape((height, width))
        b = arr[frame_number][(2 * width * height):]#.reshape((height, width))
        rgb_frame = np.stack([r, g, b])
        rgb_frame= rgb_frame.reshape((num_frames, 270,480,3))
        frames.append(rgb_frame.T)
        if frame_number%500==0:
            print('{} percent complete...'.format(frame_number*1.0/num_frames))
    return np.array(frames)'''



def group_color(logo_image, video_image, sample):
    logo_image = cv2.cvtColor(logo_image, cv2.COLOR_RGB2HSV)
    video_image = cv2.cvtColor(video_image, cv2.COLOR_RGB2HSV)
    Z = logo_image.reshape((-1,3))
    Z = np.float32(Z)

    kmeans = KMeans(n_clusters=sample)
    kmeans.fit(Z)
    labels = kmeans.predict(Z)
    center1 = kmeans.cluster_centers_
    res = center1[labels]
    res1 = res.reshape((logo_image.shape))
    res1 = res1.astype(np.uint8)
    res1 = cv2.cvtColor(res1, cv2.COLOR_HSV2RGB)
    print(center1)

    background_label = labels[0]
    # video_image = cv.GaussianBlur(video_image, (3, 3), sigmaX = 5)
    Z = video_image.reshape((-1,3))
    Z = np.float32(Z)

    labels = kmeans.predict(Z)

    # if too far, set it xto background label
    for i in range(0, len(labels)):
        if (np.linalg.norm(Z[i] - center1[labels[i]]) > 150):
            labels[i] = background_label
        # if (abs(Z[i][1] - center1[labels[i]][1]) > 50):
        #     labels[i] = background_label

    res = kmeans.cluster_centers_[labels]

    res2 = res.reshape((video_image.shape))
    res2 = res2.astype(np.uint8)
    res2 = cv2.cvtColor(res2, cv2.COLOR_HSV2RGB)
    return res1, res2

# from AffineSift import detect_logo 

# if __name__ == '__main__':
#     logo_image = read_rgb(sys.argv[1])[0]
#     # logo_image = cv.cvtColor(logo_image, cv.COLOR_RGB2GRAY)
#     logo_image = cv.GaussianBlur(logo_image, (11, 11), sigmaX = 5)
#     logo_gray = cv.cvtColor(logo_image, cv.COLOR_RGB2GRAY)

#     for i in range(39, 100):
#     # video_image = read_rgb(sys.argv[2])[0]
#         video_path = "videoSecond/" + str(i) + ".rgb"
#         print(video_path)
#         video_image = read_rgb(video_path)[0]

#         try_logo = cv.resize(logo_image, (0, 0), fx=0.3, fy=0.3)
#         res = cv2.matchTemplate(video_image,try_logo,cv2.TM_CCOEFF)
#         print(res.shape)
#         # logo_group, video_image = group_color(logo_image, video_image, 2)
#         # video_image = cv.cvtColor(video_image, cv.COLOR_RGB2GRAY)

#         # # video_image = cv.GaussianBlur(video_image, (3, 3), sigmaX = 5)
#         # video_image = cv.resize(video_image, (0, 0), fx=2, fy=2)
#         # # video_image = np.split(video_image, 3)[0]
#         # # video_image = np.split(video_image, 3, axis = 1)[0]


#         # vis, corners, pairs = detect_image(logo_gray, video_image)
#         # vis, pairs = sift_match(logo_image, video_image)
#         # cv.imwrite("ae_output/" + str(i)+ "_" + str(len(pairs)) + ".jpg", res)
#         cv.imwrite("ae_output/" + str(i) + ".jpg", res)

#         # cv.waitKey()
#     # cv.destroyAllWindows()
def template_match_test(logo_image, video_image, idx, output_dir, threshold=0.85):
    try_logo=logo_image.copy()

    img1=cv2.pyrDown(try_logo)
    img1=cv2.pyrDown(img1)

    template=cv2.pyrDown(img1)
    cv2.imshow('template',template)
    img2=video_image
    w, h = template.shape[::-1]
    # All the 6 methods for comparison in a list
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    allres=[]
    for meth in methods:
        plt.figure()

        img = img2.copy()
        method = eval(meth)

        # Apply template Matching
        res = cv2.matchTemplate(img, template, method)
        allres.append(res)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        print(min_val, min_loc, max_val, max_loc)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(img, top_left, bottom_right, 255, 2)

        plt.subplot(121), plt.imshow(res, cmap='gray')
        plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(img, cmap='gray')
        plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
        plt.suptitle(meth)

        plt.show()
def affine_size_template_match(logo_image, video_image, idx, output_dir, threshold=0.85):
    try_logo=logo_image.copy()

    local_size = [0, 0]
    local_max_val = 0
    local_max_loc = None

    for i in np.arange(4):
        try_logo = cv2.pyrDown(try_logo)
        print(try_logo.shape)
        res = cv2.matchTemplate(video_image, try_logo, cv2.TM_CCOEFF)#cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if (max_val > local_max_val):
            local_max_val = max_val
            local_size = try_logo.shape
            local_max_loc = max_loc

    top_left = local_max_loc
    bottom_right = (top_left[0] + local_size[1], top_left[1] + local_size[0])
    cv2.rectangle(video_image, top_left, bottom_right, 255, 2)

    print(local_max_val)
    if (local_max_val > 0.85):
        cv2.imwrite(output_dir + str(idx) + ".jpg", video_image)
    return local_max_val, local_max_loc


    '''logo_image = cv.GaussianBlur(logo_image, (11, 11), sigmaX = 5)

    local_size = [0, 0]
    local_max_val = 0
    local_max_loc = None
    
    for size_x in np.arange(0.05, 0.15, 0.01):
        for size_y in np.arange(0.05, 0.15, 0.01):
            try_logo = cv.resize(logo_image, (0, 0), fx=size_x, fy=size_y)
            res = cv2.matchTemplate(video_image,try_logo,cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if (max_val > local_max_val):
                local_max_val = max_val
                local_size = try_logo.shape
                local_max_loc = max_loc

    top_left = local_max_loc
    bottom_right = (top_left[0] + local_size[1], top_left[1] + local_size[0])
    cv2.rectangle(video_image, top_left, bottom_right, 255, 2)

    print(local_max_val)
    if (local_max_val > 0.85):
        cv.imwrite(output_dir + str(idx) + ".jpg", video_image)
    return local_max_val, local_max_loc'''

if __name__ == "__main__":
    logo_arr = read_rgb('/Users/kaixiwang/localUSC/CSCI576/AdProject/dataset3/Brand Images/ae_logo.rgb')[0]#read_rgb(sys.argv[1])[0]
    video_arr = read_rgb('/Users/kaixiwang/localUSC/CSCI576/AdProject/dataset3/Videos/data_test3.rgb')
    video_frame=video_arr[2430]#sys.argv[2])[0]
    logo_image = cv2.cvtColor(logo_arr, cv2.COLOR_RGB2GRAY)
    video_image = cv2.cvtColor(video_frame, cv2.COLOR_RGB2GRAY)


    affine_size_template_match(logo_image, video_image, 40,'/Users/kaixiwang/localUSC/CSCI576/AdProject/outputs/')
