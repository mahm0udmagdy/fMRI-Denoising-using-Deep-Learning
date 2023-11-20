# fMRI-Denoising using Deep Learning
 
The objective of this project is to assess the effectiveness of a deep learning technique in classifying the fMRI data by determining whether independent components (ICs) are either signal or noise.

![Alt text](model.png)

The above image clarify the model which has been used for the classification of the fMRI data. 

After the preprocessing steps, which could be done using [SPM](https://www.fil.ion.ucl.ac.uk/spm/) or [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) or both together, you can use Independant Component Analysis (ICA), which could be easily done through FSL. The ICA step will decompose your data into independent components (ICs) which are 3D Spatial maps and 1D time series, those components will be fed to the model. I also suggest to use ICA-AROMA instead of just ICA because it can also classify the noise components that is related to  motion artefacts, so you can evaluate the resuls you obtain from your model with ICA-AROMA in the processing steps. 


CNN has been used due to its ability to automatically learn discriminative features deeply embedded in the data, resulting in accurate output generation. The CNN's advantage lies in its capacity to capture high-level contextual information from both local and global perspectives, incorporating non-linear relationships. Therefore, the CNN can be employed to extract meaningful spatial and temporal features from the IC spatial maps and associated time series, respectively. 
However, I strongly encourage you to try to use LSTM with the time series data instead of CNN, It may give you even better results. 