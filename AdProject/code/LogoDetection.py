from MyImage import MyImage 
import sys
import datetime
import json
import os
import numpy as np
from scipy.io import wavfile
from datetime import datetime
import time


def recognize_logo(video_path, logo_path, method, output_dir):
    first_encounter = -1
    file = open(video_path, 'rb')
    for i in range(0, 150):
        # 10 is a Buffered time period to create more output
        if (i > first_encounter + 5 and first_encounter != -1):
            return first_encounter * 60
            
        mBytes = None
        for j in range(0, 60): # skip every 60 frame to decrease processing time
            mBytes = file.read(480 * 270 * 3)
        img = MyImage()
        img.ReadImage(mBytes)
        try:
            # max = img.TemplateMatch(i)
            if (method == "ATEMPLATE"):
                max, loc = img.ATemplate(i, logo_path, output_dir)
                print(max)
                if (max > 0.85 and first_encounter == -1):
                    first_encounter = i
                    print("Detected Logo at frame " + str(first_encounter))

            elif (method == "ASIFT"):

                found = img.ASiftMatch(i, logo_path, output_dir)
                print (found)
                if (found and first_encounter == -1):
                    first_encounter = i
                    print("Detected Logo at frame " + str(first_encounter))
        except:
            # return first_encounter
            print ("Except encountered")
            pass
    return -1

if __name__ == '__main__':
    timestamp = datetime.now().isoformat(timespec='minutes').replace(':', '').replace('T', '_')

    print(datetime.today().ctime())
    print('Running LogoDetection...')

    '''python /Users/kaixiwang/Documents/USC/CSCI-576/AdRemoval-master/LogoDetection.py /Users/kaixiwang/Documents/USC/CSCI-576/FinalProject/dataset3/Videos/data_test3.rgb /Users/kaixiwang/Documents/USC/CSCI-576/AdRemoval-master/dataset3/Videos/data_test3.wav /Users/kaixiwang/Documents/USC/CSCI-576/AdRemoval-master/config_local.json /Users/kaixiwang/Documents/USC/CSCI-576/AdRemoval-master/dataset3/outputs/video.rgb /Users/kaixiwang/Documents/USC/CSCI-576/AdRemoval-master/dataset3/outputs/audio.wav'''
    out_dir='/Users/kaixiwang/Documents/USC/CSCI-576/AdRemoval-master/output/'

    video_path = '/Users/kaixiwang/Documents/USC/CSCI-576/FinalProject/dataset3/Videos/data_test3.rgb'
    audio_path = '/Users/kaixiwang/Documents/USC/CSCI-576/AdRemoval-master/dataset3/Videos/data_test3.wav'

    config_path= '/Users/kaixiwang/Documents/USC/CSCI-576/AdRemoval-master/config_local.json'
    with open(config_path,'r') as f:
        datastore = json.load(f)
    output_video_path ='/Users/kaixiwang/Documents/USC/CSCI-576/AdRemoval-master/dataset3/outputs/video.rgb'
    output_audio_path ='/Users/kaixiwang/Documents/USC/CSCI-576/AdRemoval-master/dataset3/outputs/audio.wav'

    '''video_path = sys.argv[1]
    audio_path = sys.argv[2]

    with open(sys.argv[3], 'r') as f:
        datastore = json.load(f)

    output_video_path = sys.argv[4]
    output_audio_path = sys.argv[5]'''
    while True:
        txtlog=input('Create logfile? [y/n] ')
        print('Your choice: ', txtlog)

        if txtlog=='y':
            log_filepath=os.path.join(out_dir,"log_"+timestamp+".txt")
            print('Saving log in: \n \t{}'.format(log_filepath))
            sys.stdout = open(log_filepath, "w")
            break
        if txtlog=='n':
            break
    start=time.time()
    print(start)
    end=[]

    #if txtlog=='y':
    #    sys.stdout = open("/Users/kaixiwang/Documents/USC/CSCI-576/AdRemoval-master/output/log.txt", "w")

    for data in datastore:
        logo_path = data["logo_path"]
        print(logo_path)
        method = data["method"]

        #output_dir = data["output_dir"]
        output_dir=os.path.join(out_dir, data["output_dir"])
        if not os.path.exists(output_dir):
            print('Creating output directory: ', output_dir)
            os.makedirs(output_dir)
        print('Processing...')
        first_encounter = recognize_logo(video_path, logo_path, method, output_dir)
        print (first_encounter)
        data["first_encounter"] = first_encounter
        end.append(time.time())

    infile = open(video_path, 'rb')
    in_file_size = int(os.path.getsize(video_path) / (480 * 270 * 3)) 

    rate, audio_data1 = wavfile.read(audio_path)

    outfile = open(output_video_path, 'wb')
    idx = 0
    displace = 0
    
    audio_data = np.zeros(len(audio_data1), dtype = np.int16)
    for i in range(0, len(audio_data1)):
        if len(audio_data)==2:
            audio_data[i] = audio_data1[i][0]
        else:
            audio_data[i] = audio_data1[i]


    print(audio_data)
    for ifile_size in range(0, in_file_size):
        mBytes = infile.read(480 * 270 * 3)
        outfile.write(mBytes)
        idx += 1
        for data in datastore:
            file_size = os.path.getsize(data["ad_video"])
            frame_length = file_size / (480 * 270 * 3)
            ad_video = open(data["ad_video"], 'rb')
            if (idx == data["first_encounter"] + displace):
                rate1, data1 = wavfile.read(data["ad_audio"])
                audio_data = np.insert(audio_data, idx * 1600, data1)
                for i in range(0, int(frame_length)):
                    mBytes = ad_video.read(480 * 270 *3)
                    outfile.write(mBytes)
                    idx += 1
                displace += frame_length
    
    print(len(audio_data))

    wavfile.write(output_audio_path, rate, audio_data)
    end.append(time.time())
    for t in end:
        print(t-start)


    

