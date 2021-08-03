# MONAI-Segmentation
This is an MONAI-based segmentation for MRI-guided Prostate. This can be adapted and used for any segmentation of MRI images.

## Prerequisite
 - Python 3.0 or later
 - NVIDIA Graphics Card (RTX 1070 or later is recommended)

## Installation

Before running the scripts, make sure to install the following python libraries:
 - [MONAI](https://monai.io/)
 - [SimpleITK](https://simpleitk.readthedocs.io/en/v1.1.0/index.html) (for image conversion) (Version 2.0.0 or later)
 - tqdm (for showing a progress bar for loading images)
 - [NiBabel](https://nipy.org/nibabel/)
 - [Pytorch Version 1.9.0+111CUDA] (IF YOU HAVE A GTX 30X0 series)
 - [Pytorch Version 1.9.0+102CUDA] (IF YOU HAVE A GTX 10X0 SERIES OR LATER)

You can install all the above via PIP if you prefer.

## Training

In the following instruction, we assume that the workspace is structured as follows:
(NOTE: If you have NRRD files place them in sorted and run conversion.py)

~~~~
 + <working directory> 
     + config.ini
     + sorted_nii
         + train_images
             + Training image 1.nii.gz
             + Training image 2.nii.gz
                 ...
         + train_labels
             + Training label 1.nii.gz
             + Training label 2.nii.gz
                 ...
         + val_images
             + Validation image 1.nii.gz
             + Validation image 2.nii.gz
                 ...
         + val_labels
             + Validation label 1.nii.gz
             + Validation label 2.nii.gz
                 ...
~~~~

### Getting the code from GitHub

The script can be obtained from the GitHub repository: 

~~~~
$ cd <working directory>
$ git clone https://github.com/Luew2/MONAI-Segmentation
~~~~

### Prepare dataset

If your images are formatted in NRRD, they should be converted to Nii files, as MONAI's
image loader does not seem to handle NRRD's image header information (e.g., dimensions,
position, and orientation) correctly in the current version. The 'convert.py' script can
batch-process multiple images in a folder to convert from NRRD to Nii.

Before running the script, store the files as follows:

~~~~
 + <working directory> 
     + sorted
         + train_images
             + Training image 1.nii.gz
             + Training image 2.nii.gz
                 ...
         + train_labels
             + Training label 1.nii.gz
             + Training label 2.nii.gz
                 ...
         + val_images
             + Validation image 1.nii.gz
             + Validation image 2.nii.gz
                 ...
         + val_labels
             + Validation label 1.nii.gz
             + Validation label 2.nii.gz
                 ...
~~~~

Then, run convert.py. 
~~~~
$ cd <working directory>
$ MONAI-Segmentation/convert_dataset_to_nifty.sh
~~~~

If the script will output the images in the following directory structure:
~~~~
 + <working directory> 
     + sorted_nii
         + train_images
             + Training image 1.nii.gz
             + Training image 2.nii.gz
                 ...
         + train_labels
             + Training label 1.nii.gz
             + Training label 2.nii.gz
                 ...
         + val_images
             + Validation image 1.nii.gz
             + Validation image 2.nii.gz
                 ...
         + val_labels
             + Validation label 1.nii.gz
             + Validation label 2.nii.gz
                 ...
~~~~


### Prepare a configuration file

An example configuration file can be found in the directory cloned from the repository. Copy it to <working directory> and modify as needed.

~~~~
$ cp MONAI-Segmentation/config.sample.ini config.ini
~~~~

### Training the model

To train the model, run the following script:
~~~~
$ MONAI-Segmentation/training.py config.ini
~~~~

The result is stored in a *.pth file. The file name can be specified in config.ini using the 'model_file' parameter.

### Monitoring the training process using TensorBoard

If you have TensorBoard installed on the system, you can monitor the loss function from the web browser.
To activate it, edit the following line in the configuration file:

~~~~
[training]
use_tensorboard = 1
~~~~

Launch TensorBoard using as follows (make sure to change the current directory to where
the training script is running, as TensorBoard reads data from the file under 'runs/'):

~~~~
$ cd <working directory>
$ tensorboard --logdir=runs
~~~~

Then, open http://localhost:6006/ from a web browser.


## Inference

The trained model (*.pth) can be used for the segmentation of unseen image data (images of ice balls not used for training). If you have not trained a model, an example model file ('best_metric_model.pth') is available in the repository.

First, copy the trained model to the working directory. Assuming that the model file is named 'best_metric_model.pth':

~~~
$ cd <working directory>
$ cp <model directory>/best_metric_model.pth
~~~

Next, copy unseen images under a folder named 'sample':

~~~
$ cp <images> sample/
~~~

Make sure to have the config.ini in the working directory (see the 'Training' section), and the 'model_file' parameter matches the name of the model file. Then run the following command:

~~~~
$ MONAI-Segmentation/inference.py config.ini sample output
~~~~

The results are stored under the 'output' directory.







