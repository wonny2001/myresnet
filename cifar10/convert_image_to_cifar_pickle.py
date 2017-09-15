#python script for converting 32x32 pngs to format
from PIL import Image
import os
from array import *
import pickle
# data = array('B')

# dict = dict()
dict = {"data": [1], "filenames": "f", "labels": "l"}
# data = list()
filenames = array('B')
labels = array('B')

for dirname, dirnames, filenames in os.walk('./classes'):
    idx = 0

    for filename in filenames:
        if filename.endswith('.png'):


            im = Image.open(os.path.join(dirname, filename))
            pix = im.load()

            #print(os.path.join(dirname, filename))

            #store the class name from look at path
            class_name = int(os.path.join(dirname).split('/')[-1])
        #print class_name

        ###########################
        #get image into byte array#
        ###########################

        # create array of bytes to hold stuff
        #first append the class_name byte
        # data.append(class_name)



        #then write the rows
        #Extract RGB from pixels and append
        #note: first we get red channel, then green then blue
        #note: no delimeters, just append for all images in the set
            chars = []#array('B')

            for color in range(0,3):
                for x in range(0,32):
                    for y in range(0,32):
                        chars.append(pix[x,y][color])


            # data.append(chars)
            # filenames.append(str(filename))
            # labels.append(class_name)
            dict["data"].append(chars)
            dict["filenames"].append(filename)
            dict["labels"].append(class_name)

            # dict[idx] = {"data": data, "filenames":str(filename), "labels":class_name}
            idx += 1

############################################
#write all to binary, all set for cifar10!!#
############################################

# dict = {"data":data, "filenames":filenames, "labels":labels}
output_file = open('cifar10-ready_pickle.bin', 'wb')
# dict.tofile(output_file)
pickle.dump(dict, output_file);
output_file.close()

import pickle
import array

# img_flat = data["data"][i]
# fname = data["filenames"][i]
# label = data["labels"][i]


# for i in range(10000):

# data = {"data":[309,213,123,123], "filenames":"taewon", "labels":0}
#
# print (data["data"])
# print (data["filenames"])
# print (data["labels"])

# data = array('B')
# itemlist = ["data"]
#
# with open('outfile', 'wb') as fp:
#     pickle.dump(itemlist, fp)