cd GraphGPS
rm -rf ./datasets/processed
python convert_vi_to_vi-1.py -i 2
python main.py --cfg configs/GPS/a-mols.yaml --csv_path NHC-cracker-zzy-v2_merged.csv