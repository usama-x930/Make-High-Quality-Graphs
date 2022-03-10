print("started!")

import numpy as np
import os
import os.path
from matplotlib import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#%matplotlib inline

histdf = pd.read_csv('results.csv')
#                      ,names=['loss', 'acc', 'f1_m', 'precision_m', 'recall_m', 'top_5_categorical_accuracy',
#                             'val_loss', 'val_acc', 'val_f1_m', 'val_precision_m', 'val_recall_m', 'val_top_5_categorical_accuracy'])
histdf.head()

respath = 'images'
trans = False
sns.set_style(style="dark") # "ticks" # default style="darkgrid"



# # Plot Accuracy
############################################################
brx1 = pd.DataFrame(histdf['acc'])
lbound = histdf['acc'] - 0.03
ubound = histdf['acc'] + 0.03
smooth_path = brx1.rolling(15).mean()
path_deviation = brx1.rolling(15).std()

brx2 = pd.DataFrame(histdf['val_acc'])
lbound2 = histdf['val_acc'] - 0.03
ubound2 = histdf['val_acc'] + 0.03
smooth_path2 = brx2.rolling(15).mean()
path_deviation2 = brx2.rolling(15).std()

fig1=plt.figure(dpi=200)
plt.plot(histdf['acc'], marker='.',markerfacecolor='black') # , linestyle='.', markersize=12 , color='blue')
plt.plot(histdf['val_acc'], marker='.',markerfacecolor='black') # , linestyle='.', markersize=12 , color='blue')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
xmin, xmax, ymin, ymax = plt.axis()
plt.fill_between(path_deviation.index, lbound, ubound, color='b', alpha=.1)
plt.fill_between(path_deviation2.index, lbound2, ubound2, color='C1', alpha=.2)
fig1.savefig(respath + str(os.path.sep) +'accuracy.png',transparent=trans)
# plt.show()
# print(ymax)
             
# #
# # Plot loss
###########################################################
brx1 = pd.DataFrame(histdf['loss'])
lbound = histdf['loss'] - (max(histdf['loss'].max(), histdf['val_loss'].max()) / 20)
ubound = histdf['loss'] + (max(histdf['loss'].max(), histdf['val_loss'].max()) / 20)
smooth_path = brx1.rolling(15).mean()
path_deviation = brx1.rolling(15).std()

brx2 = pd.DataFrame(histdf['val_loss'])
lbound2 = histdf['val_loss'] - (max(histdf['loss'].max(), histdf['val_loss'].max()) / 20)
ubound2 = histdf['val_loss'] + (max(histdf['loss'].max(), histdf['val_loss'].max()) / 20)
smooth_path2 = brx2.rolling(15).mean()
path_deviation2 = brx2.rolling(15).std()

fig1=plt.figure(dpi=200)
plt.plot(histdf['loss'], marker='.',markerfacecolor='black') # , linestyle='.', markersize=12 , color='blue')
plt.plot(histdf['val_loss'], marker='.',markerfacecolor='black') # , linestyle='.', markersize=12 , color='blue')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.fill_between(path_deviation.index, lbound, ubound, color='b', alpha=.1)
plt.fill_between(path_deviation2.index, lbound2, ubound2, color='C1', alpha=.2)
fig1.savefig(respath + str(os.path.sep) + 'loss.png',transparent=trans)
# plt.show()

# # Plot f1_measure
############################################################
brx1 = pd.DataFrame(histdf['f1_m'])
lbound = histdf['f1_m'] - 0.03
ubound = histdf['f1_m'] + 0.03
smooth_path = brx1.rolling(15).mean()
path_deviation = brx1.rolling(15).std()

brx2 = pd.DataFrame(histdf['val_f1_m'])
lbound2 = histdf['val_f1_m'] - 0.03
ubound2 = histdf['val_f1_m'] + 0.03
smooth_path2 = brx2.rolling(15).mean()
path_deviation2 = brx2.rolling(15).std()

fig1=plt.figure(dpi=200)
plt.plot(histdf['f1_m'], marker='.',markerfacecolor='black') # , linestyle='.', markersize=12 , color='blue')
plt.plot(histdf['val_f1_m'], marker='.',markerfacecolor='black') # , linestyle='.', markersize=12 , color='blue')
plt.title('Model f1_measure')
plt.ylabel('f1_measure')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.fill_between(path_deviation.index, lbound, ubound, color='b', alpha=.1)
plt.fill_between(path_deviation2.index, lbound2, ubound2, color='C1', alpha=.2)
fig1.savefig(respath + str(os.path.sep) + 'f1_measure.png',transparent=trans)
# plt.show()

# # Plot precision
############################################################
brx1 = pd.DataFrame(histdf['precision_m'])
lbound = histdf['precision_m'] - 0.03
ubound = histdf['precision_m'] + 0.03
smooth_path = brx1.rolling(15).mean()
path_deviation = brx1.rolling(15).std()

brx2 = pd.DataFrame(histdf['val_precision_m'])
lbound2 = histdf['val_precision_m'] - 0.03
ubound2 = histdf['val_precision_m'] + 0.03
smooth_path2 = brx2.rolling(15).mean()
path_deviation2 = brx2.rolling(15).std()

fig1=plt.figure(dpi=200)
plt.plot(histdf['precision_m'], marker='.',markerfacecolor='black') # , linestyle='.', markersize=12 , color='blue')
plt.plot(histdf['val_precision_m'], marker='.',markerfacecolor='black') # , linestyle='.', markersize=12 , color='blue')
plt.title('Model Precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.fill_between(path_deviation.index, lbound, ubound, color='b', alpha=.1)
plt.fill_between(path_deviation2.index, lbound2, ubound2, color='C1', alpha=.2)
fig1.savefig(respath + str(os.path.sep) + 'precision.png',transparent=trans)
# plt.show()

# # Plot Recall
############################################################
brx1 = pd.DataFrame(histdf['recall_m'])
lbound = histdf['recall_m'] - 0.03
ubound = histdf['recall_m'] + 0.03
smooth_path = brx1.rolling(15).mean()
path_deviation = brx1.rolling(15).std()

brx2 = pd.DataFrame(histdf['val_recall_m'])
lbound2 = histdf['val_recall_m'] - 0.03
ubound2 = histdf['val_recall_m'] + 0.03
smooth_path2 = brx2.rolling(15).mean()
path_deviation2 = brx2.rolling(15).std()

fig1=plt.figure(dpi=200)
plt.plot(histdf['recall_m'], marker='.',markerfacecolor='black') # , linestyle='.', markersize=12 , color='blue')
plt.plot(histdf['val_recall_m'], marker='.',markerfacecolor='black') # , linestyle='.', markersize=12 , color='blue')
plt.title('Model Recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.fill_between(path_deviation.index, lbound, ubound, color='b', alpha=.1)
plt.fill_between(path_deviation2.index, lbound2, ubound2, color='C1', alpha=.2)
fig1.savefig(respath + str(os.path.sep) + 'recall.png',transparent=trans)
# plt.show()
#############################################################
############################################################



# # Plot Confusion Matrix
############################################################
cm = pd.read_csv('confusion_matrix.csv')
classes = ['CliffDiving', 'Diving']

df_cm = pd.DataFrame(cm/np.sum(cm))
df_cm.index.name = 'Predicted class'
df_cm.columns.name = 'Actual class'
plt.figure(figsize=(12,6))
plt.title('Confusion Matrix')
ax = sns.heatmap(df_cm, annot=True, fmt='.3%', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={"size": 12}) # font size
sns.set_style(style="dark")
fig = ax.get_figure()
fig.savefig(os.path.join(respath, 'confusion_matrix.png'),dpi=500, facecolor=None, edgecolor=None,
          orientation='portrait', format=None,
          transparent=True, bbox_inches='tight', pad_inches=0.1, metadata=None)



# # Plot kernel_density_estimate
############################################################
############################################################
f, axes = plt.subplots(nrows=2, ncols=5, figsize=(20,8), sharex=False, sharey=False, squeeze=True, dpi=300)
alphaval = 1
sns.set_style(style="dark") # "ticks"

ax = sns.kdeplot(histdf['loss'], shade=True, color="r", label="Loss", alpha=alphaval, ax=axes[0, 0])
ax = sns.kdeplot(histdf['acc'], shade=True, color="g", label="Accuracy", alpha=alphaval, ax=axes[0, 1])
ax = sns.kdeplot(histdf['f1_m'], shade=True, color="b", label="f1-meassure", alpha=alphaval, ax=axes[0, 2])
ax = sns.kdeplot(histdf['precision_m'], shade=True, color="m", label="Precision", alpha=alphaval, ax=axes[0, 3])
ax = sns.kdeplot(histdf['recall_m'], shade=True, color="orange", label="Recall", alpha=alphaval, ax=axes[0, 4])


ax = sns.kdeplot(histdf['val_loss'], shade=True, color="r", label="Loss", alpha=alphaval, ax=axes[1, 0])
ax = sns.kdeplot(histdf['val_acc'], shade=True, color="g", label="Accuracy", alpha=alphaval, ax=axes[1, 1])
ax = sns.kdeplot(histdf['val_f1_m'], shade=True, color="b", label="f1-meassure", alpha=alphaval, ax=axes[1, 2])
ax = sns.kdeplot(histdf['val_precision_m'], shade=True, color="m", label="Precision", alpha=alphaval, ax=axes[1, 3])
ax = sns.kdeplot(histdf['val_recall_m'], shade=True, color="orange", label="Recall", alpha=alphaval, ax=axes[1, 4])


fig = ax.get_figure()
f.savefig(respath + str(os.path.sep) + 'kernel_density_estimate.png',
            transparent=trans, bbox_inches='tight', pad_inches=0.1,dpi=300) # , facecolor=None, edgecolor=None)
# plt.show()



print("\n\n\n\n\nDone!")