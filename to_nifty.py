#! /usr/bin/python

import numpy
import glob
import argparse, sys, shutil, os, logging
import SimpleITK as sitk

def convert(input_dir, output_dir):
    
    image_files = glob.glob(os.path.join(input_dir, "*.nrrd"))
    for input_path in image_files:
        image   = sitk.ReadImage(input_path)
        intput_dir, output_file = os.path.split(input_path)
        output_file_name, output_file_ext =  os.path.splitext(output_file)
        output_file = output_file_name + '.nii.gz'
        
        output_path = os.path.join(output_dir, output_file)
        print(output_path)
        sitk.WriteImage(image, output_path)
    
    
def main(argv):

    args = []
    try:
        parser = argparse.ArgumentParser(description="Perform N4ITK Bias Correction for each coil element and combine.")
        parser.add_argument('src', metavar='SRC_DIR', type=str, nargs=1,
                            help='Source directory.')
        parser.add_argument('dst', metavar='DST_DIR', type=str, nargs=1,
                            help='Destination directory.')
        
        args = parser.parse_args(argv)
        
    except Exception as e:
        print(e)
        sys.exit()

    # input_dir = './sorted'
    # output_dir = './sorted_nii'
    input_dir = args.src[0]
    output_dir = args.dst[0]
    
    # Make the destination directory, if it does not exists.
    os.makedirs(output_dir, exist_ok=True)

    convert(input_dir, output_dir)

if __name__ == "__main__":
    main(sys.argv[1:])

