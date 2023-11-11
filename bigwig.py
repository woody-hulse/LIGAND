import pyBigWig
import os
import numpy as np

def epigenetic_data(cell_dir, chr, start, end):
    files = os.listdir(cell_dir)
    mat = np.zeros((4, end - start + 1)) # row 0-4: dnase (chromatin opening), ctcf, h3k4me, methyl

    def set_val(index, entries):
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

    for file in files:
        path = cell_dir + '/' + file
        file_obj = pyBigWig.open(path)

        if 'dnase' in file:
            vals = file_obj.values(chr, start, end)
            for i in range(len(vals)):
                if not np.isnan(vals[i]):
                    mat[0][i] = vals[i]
            continue
        elif 'ctcf' in file:
            set_val(1, file_obj.entries(chr, start, end))
        elif 'h3k4me' in file:
            set_val(2, file_obj.entries(chr, start, end))
        elif 'methyl' in file:
            set_val(3, file_obj.entries(chr, start, end))
        else:
            continue
        
        file_obj.close()

    return mat


m = epigenetic_data('GM12878','chr2',10,20)
print(m)
