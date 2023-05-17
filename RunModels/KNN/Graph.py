import numpy as np
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, xValues, yValues):
        self.xValues = xValues
        self.yValues = yValues
        self.std = []
        self.title = ""
        self.xLabel = ""
        self.yLabel = ""

    def setTitle(self, title):
        self.title = title
    
    def setXLabel(self, xLabel):
        self.xLabel = xLabel
    
    def setYLabel(self, yLabel):
        self.yLabel = yLabel

    def setStandardDev(self, std):
        self.std = std
    
    def getMean(self):
        yValueSum = sum(self.yValues)
        numInstances = len(self.yValues)
        return yValueSum/numInstances

    
    def show(self):
        plt.yticks(np.arange(0.84, 1, 0.02))
        plt.errorbar(self.xValues, self.yValues, yerr=self.std, fmt='-o')
        plt.xlabel(self.xLabel)
        plt.ylabel(self.yLabel)
        plt.title(self.title)
        plt.show()