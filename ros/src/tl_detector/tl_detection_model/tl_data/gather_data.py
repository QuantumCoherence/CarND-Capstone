import os
from subprocess import call
import glob
import ntpath

#####################################
# Short script to take data from all 4 BOSCH datasets (TL01, TL02, TL03, and TL04) 
# and combine them into a single data set maintaining direcetory structure for categories
#####################################


basepath = "TL_simulator_original"
directories = {"TL011":"TL01/data01", "TL012":"TL01/data02", "TL013":"TL01/data03", "TL02":"TL02", "TL03":"TL03","TL04":"TL04"}
destpath = "simulation_data"
categories = ["red","green","yellow","unknown"]

for dtag,dd in directories.iteritems():
  for cc in categories:
    folder = basepath + "/" + dd + "/" + cc
    if os.path.isdir(folder):
      print(folder)
      files = glob.glob("%s/*.png" % folder)
      for ff in files:
        imgname = ntpath.basename(ff)
        newimgname = destpath + "/" + cc + "/" + dtag + imgname
        cmd = "cp %s %s" % (ff,newimgname)
        print(cmd)
        call(["cp",ff,newimgname])
        


basepath = "TL_site_original"
directories = {"yuda":"yuda_data", "yuda2":"yuda2_data", "github":"github_data"}
destpath = "site_data"
categories = ["red","green","yellow","unknown"]

for dtag,dd in directories.iteritems():
  for cc in categories:
    folder = basepath + "/" + dd + "/" + cc
    if os.path.isdir(folder):
      print(folder)
      files = glob.glob("%s/*.*g" % folder)
      for ff in files:
        imgname = ntpath.basename(ff)
        newimgname = destpath + "/" + cc + "/" + dtag + imgname
        cmd = "cp %s %s" % (ff,newimgname)
        print(cmd)
        call(["cp",ff,newimgname])



