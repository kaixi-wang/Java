#!/usr/bin/env python

'''
Affine invariant feature-based image matching sample.

This sample is similar to find_obj.py, but uses the affine transformation
space sampling technique, called ASIFT [1]. While the original implementation
is based on SIFT, you can try to use SURF or ORB detectors instead. Homography RANSAC
is used to reject outliers. Threading is used for faster affine sampling.

[1] http://www.ipol.im/pub/algo/my_affine_sift/

USAGE
  asift.py [--feature=<sift|surf|orb|brisk>[-flann]] [ <image1> <image2> ]

  --feature  - Feature to use. Can be sift, surf, orb or brisk. Append '-flann'
               to feature name to use Flann-based matcher instead bruteforce.

  Press left mouse button on a feature point to see its matching point.
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv, cv2

# built-in modules
import itertools as it
from multiprocessing.pool import ThreadPool

# local modules
from common import Timer
from find_obj import init_feature, filter_matches, explore_match


def read_rgb(filename, frame_idx=None, height=270, width=480):
    print("Reading videofile...")
    f = open(filename, "r")
    arr = np.fromfile(f, dtype=np.uint8)
    num_frames=int(arr.shape[0]/(height*width*3)) #9000
    arr = arr.reshape(num_frames, (height * width * 3))
    frames=[]
    if frame_idx is None:
        frame_idx = np.arange(num_frames)
        for frame_number in frame_idx:
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
    else:
        frame_number=frame_idx
        r = arr[frame_number][0:(width * height)].reshape((height, width))
        g = arr[frame_number][(width * height):(2 * width * height)].reshape((height, width))
        b = arr[frame_number][(2 * width * height):].reshape((height, width))
        rgb_frame = np.dstack([r, g, b])
        #rgb_frame= rgb_frame.reshape((num_frames, 270,480,3))
        frames.append(rgb_frame)#.T)
    return np.array(frames)

def filter_matches_std(kp1,kp2,matches):
    distance_diff = np.abs([m[0].distance - m[1].distance for m in matches])
    min_diff = np.mean(distance_diff) + np.std(distance_diff)
    mkp1, mkp2 = [], []
    for i,m in enumerate(matches):
        if len(m) == 2 and distance_diff[i] > min_diff:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv.warpAffine(img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
    Ai = cv.invertAffineTransform(A)
    return img, mask, Ai

def sift_detect(img1,img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=2)

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

    cv2.imshow('SIFT matched', img3)
    return img3, matches, matchesMask

def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        import random
        # cv.imwrite("output/" + str(random.randint(1, 100)) + ".jpg", timg)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    print()
    return keypoints, np.array(descrs)


def detect_image(img1, img2):
    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'brisk-flann')

    detector, matcher = init_feature(feature_name)

    if img1 is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)

    if img2 is None:
        print('Failed to load fn2:', fn2)
        sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)

    print('using', feature_name)
    
    #img1 = cv.GaussianBlur(img1, (5, 5), sigmaX = 5)
    for i in np.arange(3):
        img1 = cv2.pyrDown(img1)

    pool=ThreadPool(processes = cv.getNumberOfCPUs())
    kp1, desc1 = affine_detect(detector, img1, pool=pool)
    kp2, desc2 = affine_detect(detector, img2, pool=pool)
    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

    def match_and_draw(win):
        with Timer('matching'):
            raw_matches = matcher.knnMatch(desc1, trainDescriptors = desc2, k = 2) #2
        p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
        #p1, p2, kp_pairs = filter_matches_std(kp1, kp2, raw_matches)
        if len(p1) >= 4:
            H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
            print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
            # do not draw outliers (there will be a lot of them)
            kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]

        else: #regular sift
            '''# Initiate SIFT detector
            sift = cv2.xfeatures2d.SIFT_create()

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # FLANN parameters
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary

            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(des1, des2, k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]

            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.7 * n.distance:
                    matchesMask[i] = [1, 0]

            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=2)

            img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)

            cv2.imshow('matched',img3)'''
            return sift_detect(img1,img2)

        img, corners = explore_match(win, img1, img2, kp_pairs, None, H)
        return img, corners, kp_pairs

    return match_and_draw('affine find_obj')
    # cv.waitKey()
    # print('Done')


if __name__ == '__main__':
    print(__doc__)
    import sys, getopt
    logofp=['/Users/kaixiwang/localUSC/CSCI576/AdProject/dataset3/Brand Images/ae_logo.rgb','/Users/kaixiwang/localUSC/CSCI576/AdProject/dataset3/Brand Images/hrc_logo.rgb']
    logo=1#index of logo
    pos_frames=[2370, 7020]
    img1 = read_rgb(logofp[logo])[0]#sys.argv[1])[0]
    #img1 = cv.GaussianBlur(img1, (3, 3), sigmaX = 5)
    #img1 = cv2.pyrDown(img1)

    img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    
    img2=read_rgb('/Users/kaixiwang/Documents/USC/CSCI-576/FinalProject/dataset3/Videos/data_test3.rgb',pos_frames[logofp.index(logofp[logo])])[0]
    #img2 = cv.imread(sys.argv[2])
    img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    img, corners, kp_pairs = detect_image(img1, img2)
    cv.imshow("image", img)
    cv.waitKey()
    cv.destroyAllWindows()
