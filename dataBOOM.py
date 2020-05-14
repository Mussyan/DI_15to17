import dbAffine as af
import dbZoomMove as zm
import os 

for root, dirs, files in os.walk('img'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        if((full_fname.find('ㄱ')!=-1 or full_fname.find('ㄲ')!=-1 or full_fname.find('ㄴ')!=-1 or full_fname.find('ㅏ')!=-1 or full_fname.find('ㅓ')!=-1 or full_fname.find('ㅗ')!=-1 or full_fname.find('ㅜ')!=-1) and full_fname.find('_')==-1):
            print(full_fname)
            af.runRotate(full_fname)
            zm.zoomOut(full_fname)
        
