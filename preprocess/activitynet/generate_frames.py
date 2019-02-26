# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------

import os
from util import *
import time
import json
#from joblib import delayed, Parallel
from multiprocessing import Pool

FPS = 25
EXT = '.mp4'
VIDEO_DIR = '/media/E/ActivityNet/Videos/'
video_list = os.listdir(VIDEO_DIR)
META_FILE = './activity_net.v1-3.min.json'
meta_data = json.load(open(META_FILE))
FRAME_DIR = '/media/F/ActivityNet/frames_' + str(FPS)
mkdir(FRAME_DIR)

# For parallel
file_list = []
for split in ['training', 'validation']:
    for vid, vinfo in meta_data['database'].items():
        if vinfo['subset'] == split:
            # make sure the video has been downloaded.
            vname = "v_" + vid + EXT
            if vname in video_list:
                file_list.append((vname, split, vinfo['duration']))
print("{} videos needed to be extracted".format(len(file_list)))    

def ffmpeg_extract(filename, outpath):
    status = False
    outfile = os.path.join(outpath, "image_%5d.jpg")
    command = "ffmpeg -loglevel panic -i {} -vf scale=171:128 -q:v 1 -r {} {}".format(filename, FPS, outfile)
    # hardware accelerate
    #command = "ffmpeg -loglevel panic -hwaccel cuvid -i {} -q:v 1 -r {} {}".format(filename, FPS, outfile)
    #print(command)
    try:
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, err.output	
                
    #for framename in os.listdir(outpath):
    #    resize(os.path.join(outpath, framename))
    frame_size = len(os.listdir(outpath))
        
    status = frame_size > 0
    return status, frame_size

def generage_frame_wraper(item):
    vname, split, duration = item[0], item[1], item[2]
    vid = vname[2 : -len(EXT)]
    filename = os.path.join(VIDEO_DIR, vname)
    outpath = os.path.join(FRAME_DIR, split, vid)

    if os.path.exists(outpath):
        #return True
        pass
    else:
        mkdir(outpath)            
    status, frame_size = ffmpeg_extract(filename, outpath)        
    print (filename, duration, FPS, frame_size)
    
    return status

if __name__ == '__main__':
#    file_list = file_list[:20]
    start = time.time()
    
#    for item in file_list:
#        generage_frame_wraper(item)
        
    n_jobs=16
    pool = Pool(n_jobs)
    pool.map(generage_frame_wraper, file_list)
    pool.close()
    pool.join()
    
    end = time.time()
    print("Running {} jobs, {}s per videos".format(n_jobs, (end-start)/len(file_list)))
