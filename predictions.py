from math import ceil
from random import randint, randrange
from subprocess import call
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


metrics = pd.read_csv('comp16_sz64_coralonly.csv', delimiter = ';', header = 0)
mod16_128 = pd.read_csv('comp16_sz128_coralonly.csv', delimiter = ';', header = 0)
metrics = pd.read_csv('comp8_sz128_coralonly.csv', delimiter = ';', header = 0)
#mod8_64 = pd.read_csv('comp8_sz64_coralonly.csv', delimiter = ';', header = 0)
#mod4_128 = pd.read_csv('comp4_sz128_coralonly.csv', delimiter = ';', header = 0)
#mod4_64 = pd.read_csv('comp4_sz64_coralonly.csv', delimiter = ';', header = 0)



plt.figure(1)
plt.subplot(1,3,1)
plt.plot(metrics.epoch,metrics.accuracy,label = 'train')
plt.plot(metrics.epoch,metrics.val_accuracy,label = 'test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim((0, 1))

plt.subplot(1,3,2)
plt.plot(metrics.epoch,metrics.precision,label = 'train')
plt.plot(metrics.epoch,metrics.val_precision,label = 'test')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.ylim((0, 1))

plt.subplot(1,3,3)
plt.plot(metrics.epoch,metrics.recall,label = 'train')
plt.plot(metrics.epoch,metrics.val_recall,label = 'test')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.ylim((0, 1))
plt.legend()

plt.show()

print('Final accuracy:', metrics.accuracy[14], 
'Final precision:', metrics.precision[14], 
'Final recall:', metrics.recall[14])


