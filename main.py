import numba
from scipy.stats import zscore
from scipy.signal import correlate2d , stft
from sklearn import preprocessing
import os
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import  QApplication, QMainWindow, QShortcut
from scipy.signal import spectrogram
from scipy.signal import resample
import sys
from PyQt5.QtGui import QIcon, QKeySequence
from mainwindow import Ui_MainWindow  
from pyqtgraph import PlotWidget, ROI

import numpy as np
from scipy.fftpack import rfft, rfftfreq, irfft , fft , fftfreq
from PyQt5.QtCore import pyqtSlot
import sounddevice as sd
import librosa


class MplCanvas(FigureCanvasQTAgg):
    
    def __init__(self, parent=None, width=5, height=1, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
    

 
class MyWindow(QMainWindow):   
    
    def __init__(self ):
        super(MyWindow , self).__init__()
      
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)  
        self.spectrogram_canvas = MplCanvas(self)
        self.ui.layout_spectogrm.addWidget(self.spectrogram_canvas)
        self.ui.btn_record.clicked.connect(self.record_input_voice)
        
        self.input_fs = 22050
        
        self.who_can_access = []
        
       

        self.sentense_mfcc = []

        self.sentense_spect = []


        self.user_mfcc = []
        
        self.sentenses = ["open_middle_door" , "unlock_the_gate" ,"grant_me_access"]
        self.users = [ "sara" , "reem" , "yasmeen" , "amir" , "osama" , "hesham" , "mahmoud" , "ahmed"  ]

        for user in self.users:
            user_checkbox = getattr(self.ui, f"chkBox_{user}")
            user_checkbox.stateChanged.connect(lambda state, user=f"{user}": self.users_to_access(user, state == Qt.Checked))
            
        for i ,  sentense_folder in enumerate(self.sentenses)  :
            folder_path = f"./{sentense_folder}"
            self.sentense_spect.append({sentense_folder : {}})
            for j , file_name in enumerate(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file_name)
                audio_data , sample_rate = librosa.load(file_path)
                print(audio_data.shape)
                _ , spect = self.extract_feature_points(audio_data , sample_rate)
                # dict = getattr(self , f"mfcc_{sentense_folder}")
                # dict[file_name] = spect
                self.sentense_spect[i][sentense_folder][f"{file_name.split('_')[0]}_{j}"] = spect
                # print(f"file :{file_name}")
                # print(f"file_name: {file_name}, {mfcc}")
                
            # print(self.sentense_mfcc)

        # for i ,  sentense_folder in enumerate(self.sentenses)  :
        #     folder_path = f"./{sentense_folder}"
        #     self.sentense_mfcc.append({sentense_folder : {}})
        #     for j , file_name in enumerate(os.listdir(folder_path)):
        #         file_path = os.path.join(folder_path, file_name)
        #         audio_data , sample_rate = librosa.load(file_path)
        #         mfcc = self.extract_feature_points(audio_data , sample_rate)
        #         # dict = getattr(self , f"mfcc_{sentense_folder}")
        #         # dict[file_name] = mfcc
        #         self.sentense_mfcc[i][sentense_folder][f"{file_name.split('_')[0]}_{j}"] = mfcc
        #         # print(f"file :{file_name}")
        #         # print(f"file_name: {file_name}, {mfcc}")
                
        # #     print(self.sentense_mfcc)
        
        
        
        for i , user_folder in enumerate(self.users):
            folder_path = f"./{user_folder}"
            self.user_mfcc.append({user_folder:{}})
            for j , file_name in enumerate(os.listdir(folder_path)):
                file_path = os.path.join(folder_path , file_name)
                audio_data , sample_rate = librosa.load(file_path)
                mfcc  , _ = self.extract_feature_points(audio_data , sample_rate)
                x = file_name.split('_')
                self.user_mfcc[i][user_folder][f"{file_name.split('_')[1]}_{j}"] = mfcc



        QShortcut(QKeySequence("r"), self).activated.connect(self.record_input_voice)

    
    def record_input_voice(self): # read the voice signals
        # return the audio_data
        duration = 2
        input_audio = sd.rec(int(duration * self.input_fs), samplerate=self.input_fs, channels=1, dtype=np.int16)
        input_audio = np.squeeze(input_audio)
        self.ui.btn_record.setText("recording")
        # self.print_access_or_denied()
        sd.wait()
        self.plot_spectogram(input_audio)
        
        # get mfcc 
        mfcc , spect = self.extract_feature_points(input_audio , self.input_fs)
        # spect = self.calc_spect(input_audio , self.input_fs)

        # print(f"input mfcc :{mfcc.shape}")
        user_prob = self.featurepoints_corrlation_for_users(mfcc)
        # sent_prob = {}
        sent_prob = self.featurepoints_corrlation_for_sentences_spect(spect)
        # sent_prob = self.featurepoints_corrlation_for_sentences(mfcc)
        
        socre_sent , sent = self.get_score_from_dict(sent_prob)
        socre_user , who = self.get_score_from_dict(user_prob)
        
        self.print_access_or_denied(socre_sent, sent ,  socre_user ,who )
        
        # if socre_sent > 0.35 and socre_user > 0.35:
            
            
        
        
        self.print_sentense_scores(sent_prob , user_prob)
     
        

        
        
    
    
    def plot_spectogram(self , audio_data): # draw the spectogram of the input voice 
        self.spectrogram_canvas.axes.clear()
        self.spectrogram_canvas.axes.specgram(audio_data, Fs=self.input_fs)
        self.spectrogram_canvas.draw()
        
        
       

    def extract_feature_points(self , audio_data , sample_rate): # get the feature points from the spectogram
        audio_data = audio_data.astype(np.float32)
        # mfcc = np.mean(librosa.feature.mfcc(y = audio_data , sr = sample_rate , n_mfcc=50), axis=0)
        mfcc = (librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=50))
        mfcc = mfcc.T
        scaler = preprocessing.StandardScaler()
        mfcc_normalized = scaler.fit_transform(mfcc)
        mfcc_normalized = mfcc_normalized.T

        stft = np.abs(librosa.stft(audio_data))
        stft_db = librosa.amplitude_to_db(np.abs(stft))
        # spectrogram = librosa.feature.melspectrogram(y = audio_data, sr=sample_rate) 
        spectrogram = librosa.feature.melspectrogram(S = stft_db, sr=sample_rate) 
        spectrogram = spectrogram.T
        spectrogram_normalized = scaler.fit_transform(spectrogram)
        spectrogram_normalized = spectrogram_normalized.T
        
        return mfcc_normalized.flatten() , spectrogram_normalized.flatten()
    
    # @numba.jit(parallel=True)
    def featurepoints_corrlation_for_sentences_spect(self , input_specto):
        max_corr_dict = {}

        for i ,sentense in enumerate(self.sentense_spect):
            max_corr_for_each_sent = 0
            for user , spect in sentense[self.sentenses[i]].items():
                corr_arr = np.correlate(input_specto, spect, mode='full')
                max_corr_position = np.unravel_index(np.argmax(corr_arr), corr_arr.shape)
                # corr = corr_arr[max_corr_position]
                corr = np.max(corr_arr)

                corr_arr__ = np.correlate(spect, spect, mode='full')
                max_corr_position__ = np.unravel_index(np.argmax(corr_arr__), corr_arr__.shape)
                # corr__ = corr_arr__[max_corr_position__]
                corr__ = np.max(corr_arr__)


                similarity_score = corr / corr__

                
                if similarity_score > max_corr_for_each_sent:
                    max_corr_for_each_sent = similarity_score

                print(f"{sentense.keys()} :{user} : {similarity_score}")
            max_corr_dict[f"{list(sentense.keys())[0]}"] = max_corr_for_each_sent

            print(f"max = {max_corr_for_each_sent}")
            print("_________")
        print(max_corr_dict)
       
        return max_corr_dict
        
        
    
    def featurepoints_corrlation_for_users(self , input_mfcc):
        max_corr_dict = {}
        for i ,user in enumerate(self.user_mfcc):
            max_corr_for_each_user = 0
            for sentense , mfcc in user[self.users[i]].items():
                # print(user)
                corr_arr = np.correlate(input_mfcc, mfcc, mode='full')
                max_corr_index = np.argmax(corr_arr)
                # corr = np.max(corr_arr)
                corr = corr_arr[max_corr_index]
                
                corr_arr__ = np.correlate(mfcc, mfcc, mode='full')
                max_corr_index__ = np.argmax(corr_arr__)
                corr__ = corr_arr__[max_corr_index__]
                # corr__ = np.max(corr_arr__)
                similarty_score = corr/corr__
                
                
                if similarty_score > max_corr_for_each_user:
                    max_corr_for_each_user = similarty_score

                print(f"{user.keys()} :{sentense} : {similarty_score}")
                # print(f"{user.keys()} :{user} : {corr__}")
                # print(f"{user.keys()} :{user} : {corr}")
            max_corr_dict[f"{list(user.keys())[0]}"] = max_corr_for_each_user

            # print(f":{user} : {tot_corr / num}")
            print(f"max = {max_corr_for_each_user}")
            print("_________")
        print(max_corr_dict)
        for x , y in max_corr_dict.items():
            print(x)
            print(y)
                
       
        return max_corr_dict
    
        pass

    def print_sentense_scores(self , probs_sent , prop_user): # see how close the input signal and to the 3 sentences or the 8 user voices
        for sentsnce , prob in probs_sent.items():
            cell = getattr(self.ui , f"lbl_prop_{sentsnce }")
            cell.setStyleSheet("")
            if prob > .72:
                cell.setStyleSheet("color: green")
                cell.setText(str(round(prob, 2)))
                continue
                
            cell.setText(str(round(prob , 2)))
            
        for user , prob in prop_user.items():
            cell = getattr(self.ui , f"lbl_prob_{user }")
            cell.setStyleSheet("")
            if prob > .37:
                cell.setStyleSheet("color: green")
                cell.setText(str(round(prob , 2)))
                continue
                
            cell.setText(str(round(prob, 2)))

        
        # pass
    
    
    def users_to_access(self , user , ischecked): # update how can access from the users 
        print(ischecked)
        if ischecked:
            self.who_can_access.append(user)
        else:
            self.who_can_access.remove(user)
            
        print(self.who_can_access)


    def print_access_or_denied(self , max_sent ,sent, max_user ,who): #print if the user is allowed to access or not 
        if max_sent > 0.72 and max_user > 0.37 and who in self.who_can_access:
            self.ui.lbl_access.setText(f'<font color="green">welcome {who}</font>')
            

        else:
            self.ui.lbl_access.setText('<font color="red">Access Denied</font>')
                
    def get_score_from_dict(self , dict):
        max = 0
        for _ , score in dict.items():
            if score > max :
                max = score
                user_sent = _
        return max , user_sent

        
        
    

def main():
    app = QApplication(sys.argv)
    window = MyWindow() 
   
   
    window.showMaximized()
    window.show()
    
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()