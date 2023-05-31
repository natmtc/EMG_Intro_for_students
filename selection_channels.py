from Prepro_EMG import *
from Prepro_filters import *
import glob, os
import pandas as pd

######################################## SELECT CHANNELS THAT WE WANT ACROSS GRIDS #######################
def select_channels(letters, fsampl, bin_size):
    #This function takes the EMG channels relative to each letter from the Sign Language Dataset 
    # and finds the channels that are noisy (by doing a correlation analysis)
    #then adds in a dataframe stucture the channels that are noisy for each grid
    emg_obj = offline_EMG('/Users/natal/Desktop/2022-2023/PhD_Q1/Natalia/EMG',0)#import the EMG files
    discard_channels = {'A': [[],[],[]], 'B': [[],[],[]],'C': [[],[],[]],'D': [[],[],[]],'E': [[],[],[]],'F': [[],[],[]],'G': [[],[],[]],'H': [[],[],[]],'I': [[],[],[]],'J': [[],[],[]],'K': [[],[],[]],'L': [[],[],[]],'M': [[],[],[]],'MY': [[],[],[]],'N': [[],[],[]],'O': [[],[],[]],'P': [[],[],[]],'Q': [[],[],[]],'R': [[],[],[]],'S': [[],[],[]]}
    discard_channels = pd.DataFrame(data=discard_channels) #create a dataframe for adding the channels that we will need to remove

    for index in range(len(letters)): #take all the letters that have a dataset
        all_files = "./"+letters[index]+ ".otb+" #open OTB file
        emg_obj.open_otb(all_files) # adds signal_dict to the emg_obj
        letter = all_files[2]
        if all_files[3]=='Y':
            letter= 'MY'

        #INFORMATION ABOUT THE DATASET
        # print(f"Number of channels: {emg_obj.signal_dict['nchans']}")  #chanels-by-time
        # print(f"Number of grids: {emg_obj.signal_dict['ngrids']}")  #chanels-by-time
        # print(f"Names of the grids: {emg_obj.signal_dict['grids']}")  
        # print(f"Names of the muscles: {emg_obj.signal_dict['muscles']}")  
        # print(f"Fsamp: {emg_obj.signal_dict['fsamp']}")  

        #FILTERING
        signal1= notch_filter(emg_obj.signal_dict['data'],emg_obj.signal_dict['fsamp'])
        signal2=bandpass_filter(signal1,10240,emg_type =0)

        # FIND NOISY CHANNELS BY DOING A CORRELATION ANALYSIS
        
        grid1= np.abs(signal2[:64,:])
        grid1_corr=[]
        for i in range(grid1.shape[0]):
            if i==0:  #first we need to compute the mean of the rest of the channels to be able to compare with channel i
                mean_channels= np.mean(grid1[i+1:,:],axis=0)
            else:
                signal_conc= np.concatenate((grid1[:i,:],grid1[i+1:,:]))
                mean_channels= np.mean(signal_conc, axis=0)

            corr= np.correlate(grid1[i,:], mean_channels) #correlation channel i with avg channels -{i}
            grid1_corr.append(corr)

        norm = np.linalg.norm(grid1_corr) 
        normal_array1 = grid1_corr/norm #normalise the correlation data

        #####
        grid2= np.abs(signal2[129:192,:])
        grid2_corr=[]
        for i in range(grid2.shape[0]):
            if i==0:
                mean_channels= np.mean(grid2[i+1:,:],axis=0)
            else:
                signal_conc= np.concatenate((grid2[:i,:],grid2[i+1:,:]))
                mean_channels= np.mean(signal_conc, axis=0)

            corr= np.correlate(grid2[i,:], mean_channels)

            grid2_corr.append(corr)

        norm = np.linalg.norm(grid2_corr)
        normal_array2 = grid2_corr/norm

        #####
        grid3= np.abs(signal2[193:256,:])
        grid3_corr=[]
        for i in range(grid3.shape[0]):
            if i==0:
                mean_channels= np.mean(grid3[i+1:,:],axis=0)
            else:
                signal_conc= np.concatenate((grid3[:i,:],grid3[i+1:,:]))
                mean_channels= np.mean(signal_conc, axis=0)

            corr= np.correlate(grid3[i,:], mean_channels)

            grid3_corr.append(corr)

        norm = np.linalg.norm(grid3_corr)
        normal_array3 = grid3_corr/norm

        ##### COMPUTE DEVIATIONS
        deviations1 = [(x - normal_array1.mean()) ** 2 for x in normal_array1]
        deviations2 = [(x - normal_array2.mean()) ** 2 for x in normal_array2]
        deviations3 = [(x - normal_array3.mean()) ** 2 for x in normal_array3]



        #FIND THE NOISY CHANNELS USING STD 
        bad_chans1= []
        for i in range(grid1.shape[0]):
            if deviations1[i]>0.002:
                bad_chans1.append(i)
        discard_channels.iloc[0,index]= bad_chans1

        bad_chans2= []
        for i in range(grid2.shape[0]):
            if deviations2[i]>0.002:
                bad_chans2.append(i)
        discard_channels.iloc[1,index]= bad_chans2

        bad_chans3= []
        for i in range(grid3.shape[0]):
            if deviations3[i]>0.002:
                bad_chans3.append(i)
        discard_channels.iloc[2,index]= bad_chans3

        # CREATE THE EMG RATES
        time_points= grid1.shape[1]/fsampl #the same for the three grids so doesn't matter which one to take
        eq_bin_size= bin_size*grid1.shape[1]/time_points #we need to create a timeline

        periods= list(range(0, grid1.shape[1], np.int64(np.round(eq_bin_size))))
        rate_grid1= np.zeros((grid1.shape[0],len(periods)))
        for i in range(grid1.shape[0]):
            for j in range(len(periods)-1):
                mean_bin_size=np.mean(np.abs(grid1[i,periods[j]:periods[j+1]]))
                rate_grid1[i,j]= mean_bin_size

        rate_grid2= np.zeros((grid2.shape[0],len(periods)))
        for i in range(grid2.shape[0]):
            for j in range(len(periods)-1):
                mean_bin_size=np.mean(np.abs(grid2[i,periods[j]:periods[j+1]]))
                rate_grid2[i,j]= mean_bin_size

        rate_grid3= np.zeros((grid3.shape[0],len(periods)))
        for i in range(grid3.shape[0]):
            for j in range(len(periods)-1):
                mean_bin_size=np.mean(np.abs(grid3[i,periods[j]:periods[j+1]]))
                rate_grid3[i,j]= mean_bin_size


        value= ('rate'+letter+'_grid1.csv')
        np.savetxt(value,rate_grid1, fmt="%f", delimiter=',')

        value= ('rate'+letter+'_grid2.csv')
        np.savetxt(value,rate_grid2, fmt="%f", delimiter=',')

        value= ('rate'+letter+'_grid3.csv')
        np.savetxt(value,rate_grid3, fmt="%f", delimiter=',')


    discard_channels.to_csv('chan_to_rej.csv', index=False)
