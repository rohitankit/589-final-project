import numpy as np
import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.title = ""
        self.xLabel = ""
        self.yLabel = ""
    
    def plot(self, xValues, yValues, label):
        plt.plot(xValues, yValues, label=label)

    def setTitle(self, title):
        self.title = title
    
    def setXLabel(self, xLabel):
        self.xLabel = xLabel
    
    def setYLabel(self, yLabel):
        self.yLabel = yLabel
    
    def show(self):
        plt.xlabel(self.xLabel)
        plt.ylabel(self.yLabel)
        plt.title(self.title)
        plt.legend()
        plt.show()