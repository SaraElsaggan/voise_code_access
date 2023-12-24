
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
    
        QShortcut(QKeySequence("Ctrl+o"), self).activated.connect(self.read_voice)

    
    def read_voice(self): # read the voice signals
        self.file_path , _ = QFileDialog.getOpenFileName(self, "Open file", "~")
        self.sample_rate, self.original_sig = wavfile.read(self.file_path)
        self.spectrogram_canvas.axes.clear()
        self.spectrogram_canvas.axes.specgram(self.original_sig , Fs = self.sample_rate)
        self.spectrogram_canvas.draw()
    
    def calc_voice_spectogram(self): # calculate the spectogram of the voice signal
        pass
    
    def extract_feature_points(self): # get the feature points from the spectogram
        pass
    
    def featurepoints_corrlation(self): #compare between the inpus voice signal and the feature point from other spectograms
        pass
    
    def calc_scores(self): # see how close the input signal and to the 3 sentences or the 8 user voices
        pass
    
    def plot_input_spectogram(self): # just draw the spectogram of the input voice 
        pass
    
    def users_to_acess(self): # update how can access from the users
        pass
    
    def print_acess_or_denied(self): #print if the user is allowed to access or not 
        pass
    

def main():
    app = QApplication(sys.argv)
    window = MyWindow() 
   
   
    window.showMaximized()
    window.show()
    
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()