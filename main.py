from scipy.signal import correlate2d
from sklearn import preprocessing
import os
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import  QApplication, QMainWindow, QShortcut, QFileDialog , QSplitter , QFrame , QSlider
from scipy.signal import spectrogram
from scipy.signal import resample
import sys
from PyQt5.QtGui import QIcon, QKeySequence
from mainwindow import Ui_MainWindow  
from pyqtgraph import PlotWidget, ROI

import numpy as np
import pandas as pd
from scipy.io import wavfile
import pyqtgraph as pg
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
        # self.layout_prob_sentence = MplCanvas(self)
        # self.layout_prob_user = MplCanvas(self)
        self.spectrogram_canvas = MplCanvas(self)
        self.ui.layout_spectogrm.addWidget(self.spectrogram_canvas)
        # self.ui.layout_prob_sentence_2.addWidget(self.layout_prob_sentence)
        # self.ui.layout_prob_user_2.addWidget(self.layout_prob_user) 
        
        self.ui.btn_record.clicked.connect(self.record_input_voice)
        
        self.input_fs = 22050
        # self.input_fs = 44100
        self.access = False
        self.who_can_access = []
        
        for i in range(8):
            user_checkbox = getattr(self.ui, f"chkBox_user_{i+1}")
            # user_checkbox.stateChanged.connect(lambda i=i: self.users_to_acess(f"user_{i+1}", user_checkbox.isChecked()))
            user_checkbox.stateChanged.connect(lambda state, user=f"user_{i+1}": self.users_to_acess(user, state == Qt.Checked))

        self.sentenses_mfcc = {}
        # relative_path = os.path.join("project", "voices")

        # # Construct the absolute path
        # absolute_path = os.path.abspath(relative_path).replace("\\", "/")
        # self.voice_folder_path = absolute_path
        self.voice_folder_path = "C:/Users/Sara/Desktop/sara_voice_code_access/voices"
        print(self.voice_folder_path)
        for i,  file_name in enumerate(os.listdir(self.voice_folder_path)):
            file_path = os.path.join(self.voice_folder_path, file_name)
            audio_data , sampleRate = librosa.load(file_path)
            print(sampleRate*2)
            # mfcc = self.extract_feature_points(audio_data , sampleRate*2) # multblied by 2 coz the return sample rate of the audio is 22050 
            mfcc = self.extract_feature_points(audio_data , sampleRate)
            self.sentenses_mfcc[file_name] = mfcc
            print(f"file{i} :{mfcc.shape}")
            




        QShortcut(QKeySequence("r"), self).activated.connect(self.record_input_voice)

    
    def record_input_voice(self): # read the voice signals
        # return the audio_data
        duration = 5
        input_audio = sd.rec(int(duration * self.input_fs), samplerate=self.input_fs, channels=1, dtype=np.int16)

        self.print_acess_or_denied()
        sd.wait()
        self.plot_spectogram(input_audio)
        mfcc = self.extract_feature_points(input_audio , self.input_fs)
        print(f"input :{mfcc.shape}")
        
        self.featurepoints_corrlation(mfcc)


        
        self.access = True  # here i will use the model to predict if the acess is denied or not
        # self.print_acess_or_denied()
        
        return input_audio
    
    
    def plot_spectogram(self , audio_data): # draw the spectogram of the input voice 
        self.spectrogram_canvas.axes.clear()
        self.spectrogram_canvas.axes.specgram(audio_data[:, 0], Fs=self.input_fs)
        self.spectrogram_canvas.draw()
        
        
        
    '''mode 1 using mfcc'''
    def extract_feature_points(self , audio_data , sample_rate): # get the feature points from the spectogram
        audio_data = audio_data.astype(np.float32)
        # mfcc = np.mean(librosa.feature.mfcc(y = audio_data , sr = sample_rate , n_mfcc=50), axis=0)
        mfcc = (librosa.feature.mfcc(y=audio_data.flatten(), sr=sample_rate, n_mfcc=50))

        mfcc = mfcc.T
        scaler = preprocessing.StandardScaler()
        mfcc_normalized = scaler.fit_transform(mfcc)

        # Transpose the matrix back to the original shape
        mfcc_normalized = mfcc_normalized.T

        # mfcc = mfcc.flatten()
        # print(f"")

        return mfcc_normalized.flatten()
        # pass
    
    def featurepoints_corrlation(self , mfcc ): #compare between the inpus voice signal and the feature point from other spectograms
        # correlation = np.corrcoef(self.sentenses_mfcc['test.wav'], mfcc)[0, 1] 
        # print(f"correlation{correlation}")
        
        
        correlation = np.corrcoef(self.sentenses_mfcc['open_middle_door.wav'], mfcc)[0, 1] 
        print(f"correlation{correlation}")
        correlation = np.corrcoef(self.sentenses_mfcc['grant_me_access.wav'], mfcc)[0, 1] 
        print(f"correlation{correlation}")
        correlation = np.corrcoef(self.sentenses_mfcc['unlock_the_gate.wav'], mfcc)[0, 1] 
        print(f"correlation{correlation}") # till here not cross corrlation





        # correlation = np.corrcoef(self.sentenses_mfcc['grant_me_access.wav'], self.sentenses_mfcc['grant_me_access.wav'])[0, 1]
        # correlation_1 = np.correlate(self.sentenses_mfcc['grant_me_access.wav'], mfcc, mode='full')  #crosse corr
        # correlation = correlate2d(self.sentenses_mfcc['grant_me_access.wav'], mfcc, mode='full') #crosse corr
        # correlation = np.correlate(self.sentenses_mfcc['grant_me_access.wav'], mfcc, mode='full') #crosse corr
        # max_corr_index = np.argmax(correlation)
        
        # # The similarity score is the maximum value of the cross-correlation
        # similarity_score = correlation[max_corr_index]

        # print(correlation_1 == correlation)
        # correlation = np.corrcoef(self.sentenses_mfcc['grant_me_access.wav'], mfcc)[0, 1]
        # print(f"correlation{correlation}")
        # print(f"correlation{correlation}")
        # print(correlation)

        pass
    
    def calc_scores(self): # see how close the input signal and to the 3 sentences or the 8 user voices
        pass
    
    
    def users_to_acess(self , user , ischecked): # update how can access from the users 
        print(ischecked)
        if ischecked:
            self.who_can_access.append(user)
        else:
            self.who_can_access.remove(user)
            
        print(self.who_can_access)


    def print_acess_or_denied(self ): #print if the user is allowed to access or not 
        if self.access :
            self.ui.lbl_access.setText('<font color="green">Access Granted</font>')
        else:
            self.ui.lbl_access.setText('<font color="red">Access Denied</font>')
            

        
    

def main():
    app = QApplication(sys.argv)
    window = MyWindow() 
   
   
    window.showMaximized()
    window.show()
    
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()