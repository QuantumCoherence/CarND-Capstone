import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

label_color = {0:'red',1:'orange',2:'green',3:'unknown'}

num_images = 635

startimg = 0

# for labelling
# for ii in range(startimg,num_images):
# 	labname = "/home/philip/sdc/CarND-Capstone/ros/src/tl_detector/tl_labels/tl_label_%05d.txt" % ii

# 	with open(labname, "r") as label_file:
# 		label = np.int(label_file.read())

# 	print("{} {} {}".format(ii,label_color[label],label))

# 	resp = raw_input("enter label :")

# 	if(resp):
# 		with open(labname, "w") as label_file:
# 			label_file.write("%d" % np.int(resp))



for ii in range(startimg,num_images):
	labname = "/home/philip/sdc/CarND-Capstone/ros/src/tl_detector/tl_labels/tl_label_%05d.txt" % ii

	imgname = "/home/philip/sdc/CarND-Capstone/ros/src/tl_detector/tl_images/tl_image_%05d.png" % ii

	with open(labname, "r") as label_file:
		label = np.int(label_file.read())

	print("{} {}".format(ii,label_color[label]))
	img=mpimg.imread(imgname)

	fig, ax = plt.subplots(figsize=(32,24))
	ax.imshow(img)
	plt.title("%d Label : %d %s" % (ii,label,label_color[label]),fontsize=60)
	plt.draw()

	response = plt.waitforbuttonpress() # this will wait for indefinite time
	
	plt.close()
	
	
	
	


