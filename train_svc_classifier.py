from utils import classifier as sc
import argparse
import os 
'''
	Perform training the SVC classifier in the end of recognition.
'''
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dir", type=str, required=True,
	help="path to the root of data")
ap.add_argument("-o", "--outdir", type=str, required=False, default='Model',
	help="path to where model and files to be saved")
args = vars(ap.parse_args())

print(f"Data directory - {args['dir']}")
print(f"output directory - {args['outdir']}")

sc.train_classifier(data_dir=args['dir'],
                    export_dir=args['outdir'])
