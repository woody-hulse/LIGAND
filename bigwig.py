import pyBigWig
import os
import numpy as np

def epigenetic_data(cell_dir, chr, start, end):
    files = os.listdir(cell_dir)
    mat = np.zeros((4, end - start + 1)) # row 0-4: dnase (chromatin opening), ctcf, h3k4me, methyl

    def set_val_bigbed(index, entries):
        if not entries:
            mat[index,0:5] = 7
            return
        for e in entries:
            region_start = e[0]
            region_end = e[1]
            string_vals = e[2].split('\t')
            val = 0 
            if string_vals[0].isnumeric():
                val = float(string_vals[0])/1000
            else:
                val = float(string_vals[1])/1000
            mat[index,region_start-start:region_end-start] = val
    
    def set_val_bigwig(index, vals):
        for i in range(len(vals)):
                if not np.isnan(vals[i]):
                    mat[index][i] = vals[i]
    index = 0
    for file in files:
        path = cell_dir + '/' + file
        print(path)
        file_obj = pyBigWig.open(path)

        if 'dnase' in file: index = 0
        elif 'ctcf' in file: index = 1
        elif 'h3k4meta' in file: index = 2
        elif 'methyl' in file: index = 3
        else: continue
        
        if 'bigWig' in file:
            set_val_bigwig(index, file_obj.values(chr, start, end))
        if 'bigBed' in file:
            set_val_bigbed(index, file_obj.entries(chr, start, end))
        file_obj.close()

    return mat

m = epigenetic_data('GM12878','chr2',10,20)
print(m)
