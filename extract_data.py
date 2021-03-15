# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 18:24:16 2020

@author: IniLaptop
"""
import json
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import librosa

DATASET_PATH = "./dataset"
JSON_PATH = "Extracted_data.json"
SAMPLE_RATE = 44100
TRACK_DURATION = 1 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_mfcc(dataset_path, json_path, num_mfcc=40, n_fft=2048, hop_length=512, num_segments=10):

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                #load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # WAVEFORM
                # display waveform
                plt.figure(figsize=(15,4))
                plt.plot(np.linspace(0, len(signal) / sample_rate, num=len(signal)),signal)
                plt.grid(True)
                plt.show()
               
                
                # FFT -> power spectrum
                # perform Fourier transform
                fft = np.fft.fft(signal)
                # calculate abs values on complex numbers to get magnitude
                magnitude = np.abs(fft)
                
                # create frequency variable
                f = np.linspace(0,sample_rate, len(magnitude))
                
                # take half of the spectrum and frequency
                left_magnitude = magnitude[:int(len(f)/2)]
                left_f = f[:int(len(f)/2)]
                
                plt.plot(left_f, left_magnitude)
                plt.xlabel("frequency")
                plt.ylabel("Magnitude")
                plt.show()
                
                
                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(signal[start:finish], 
                                                sample_rate, 
                                                n_mfcc=num_mfcc, 
                                                n_fft=n_fft, 
                                                hop_length=hop_length,
                                                lifter=10)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, d+1))
                        
                
                librosa.display.specshow(mfcc.T, sr = sample_rate, hop_length= hop_length)
                plt.xlabel("Time")
                plt.ylabel("mfcc")
                plt.colorbar()
                plt.show()
                
    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    begin = time.time() 
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)
    time.sleep(1) 
    # store end time 
    end = time.time() 
      
    # total time taken 
    print(f"\nTotal runtime of the program is {end - begin} sec")