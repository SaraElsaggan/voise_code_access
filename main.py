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
        
        self.input_fs = 44100
        self.access = False
        self.who_can_access = []
        
        for i in range(8):
            user_checkbox = getattr(self.ui, f"chkBox_user_{i+1}")
            # user_checkbox.stateChanged.connect(lambda i=i: self.users_to_acess(f"user_{i+1}", user_checkbox.isChecked()))
            user_checkbox.stateChanged.connect(lambda state, user=f"user_{i+1}": self.users_to_acess(user, state == Qt.Checked))



        QShortcut(QKeySequence("Ctrl+r"), self).activated.connect(self.record_input_voice)

    
    def record_input_voice(self): # read the voice signals
        duration = 5 
        self.input_audio = sd.rec(int(duration * self.input_fs), samplerate=self.input_fs, channels=1, dtype=np.int16)
        sd.wait()
        self.plot_spectogram()
        self.access = False # here i will use the model to predict if the acess is denied or not
        self.print_acess_or_denied()
    
    def plot_spectogram(self): # draw the spectogram of the input voice 
        self.spectrogram_canvas.axes.clear()
        self.spectrogram_canvas.axes.specgram(self.input_audio[:, 0], Fs=self.input_fs)
        self.spectrogram_canvas.draw()
        
        
    
    def extract_feature_points(self): # get the feature points from the spectogram
        pass
    
    def featurepoints_corrlation(self): #compare between the inpus voice signal and the feature point from other spectograms
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