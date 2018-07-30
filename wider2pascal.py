'''
    WIDER Faces to Pascal VOC:
    Adapted from the Matlab version (https://github.com/albanie/wider2pascal)
'''
import os
import scipy.io as sio
import xml.etree.cElementTree as ET
import numpy as np
import cv2
from shutil import copyfile

def generateXML(imgDir, imgName, bboxList, targetName):
	# info
	objectName = "face"
	imgFolder = "WIDER"
	databaseName = "WIDER FACE"
	annotationType = "WIDER"
	ownerName = " Multimedia Laboratory, Department of Information Engineering, The Chinese University of Hong Kong (http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/)"

	# annotation
	anno = ET.Element("annotation")

	# folder 
	folder = ET.SubElement(anno, "folder")
	folder.text = imgFolder

	# filename
	filename = ET.SubElement(anno,"filename")
	filename.text = imgName

	# source
	source = ET.SubElement(anno,"source")
	database = ET.SubElement(source, "database")
	database.text = databaseName
	db_anno = ET.SubElement(source, "annotation")
	db_anno.text = annotationType

	# size
	size = ET.SubElement(anno, "size")
	# get metadata from image
	# shape = (rows, cols, channels)
	immeta = cv2.imread(imgDir)
	# size->width 
	width = ET.SubElement(size, "width")
	width.text = str(immeta.shape[1])
	# size->height
	height = ET.SubElement(size, "height")
	height.text = str(immeta.shape[0])
	# size->depth
	depth = ET.SubElement(size, "depth")
	depth.text = str(immeta.shape[2])

	# owner
	owner = ET.SubElement(anno, "owner")
	name = ET.SubElement(owner, "name")
	name.text = ownerName

    # bounding box list
	if bboxList.ndim == 1:
		xmin = bboxList[0]
		ymin = bboxList[1]
		xmax = bboxList[0] + bboxList[2]
		ymax = bboxList[1] + bboxList[3]

		# object
		obj = ET.SubElement(anno, "object")

		name = ET.SubElement(obj,"name")
		name.text = objectName

		bbox = ET.SubElement(obj,"bndbox")

		elem = ET.SubElement(bbox, "xmin")
		elem.text = str(round(xmin))

		elem = ET.SubElement(bbox, "ymin")
		elem.text = str(round(ymin))

		elem = ET.SubElement(bbox, "xmax")
		elem.text = str(round(xmax))

		elem = ET.SubElement(bbox, "ymax")
		elem.text = str(round(ymax))
	else:
		for i in range(len(bboxList)):
			xmin = bboxList[i][0]
			ymin = bboxList[i][1]
			xmax = bboxList[i][0] + bboxList[i][2]
			ymax = bboxList[i][1] + bboxList[i][3]

			# object
			obj = ET.SubElement(anno, "object")

			name = ET.SubElement(obj,"name")
			name.text = objectName

			bbox = ET.SubElement(obj,"bndbox")

			elem = ET.SubElement(bbox, "xmin")
			elem.text = str(round(xmin))

			elem = ET.SubElement(bbox, "ymin")
			elem.text = str(round(ymin))

			elem = ET.SubElement(bbox, "xmax")
			elem.text = str(round(xmax))

			elem = ET.SubElement(bbox, "ymax")
			elem.text = str(round(ymax))

	tree = ET.ElementTree(anno)
	tree.write(targetName)


def generateAnnotations(widerRootDir, annotationsDir, mode='train'):
	
	# check if destination exists
	if not os.path.exists(annotationsDir):
		os.makedirs(annotationsDir)

	# read matlab file 
	filepath = os.path.join(widerRootDir, 'wider_face_split')
	filename = os.path.join(filepath,'wider_face_'+mode+'.mat')
	data = sio.loadmat(filename, squeeze_me=True)

	for eventId in range(len(data['event_list'])):
		files 	= data['file_list'][eventId]
		bboxList= data['face_bbx_list'][eventId]
		
		for j in range(len(files)):
			imgName = str(files[j])+'.jpg'
			imgDir = os.path.join(widerRootDir, 'WIDER_'+str(mode),'images',data['event_list'][eventId], imgName)
			annotationFilename = os.path.join(annotationsDir,str(files[j])+'.xml')

			generateXML(imgDir, imgName, bboxList[j], annotationFilename)


def copyImages(widerRootDir, jpegImagesDir, mode='train'):
	# check if destination exists
	if not os.path.exists(jpegImagesDir):
		os.makedirs(jpegImagesDir)

	# read matlab file 
	filepath = os.path.join(widerRootDir, 'wider_face_split')
	filename = os.path.join(filepath,'wider_face_'+mode+'.mat')
	data = sio.loadmat(filename, squeeze_me=True)

	for eventId in range(len(data['event_list'])):
		files 	= data['file_list'][eventId]
		for j in range(len(files)):
			imgName = str(files[j])+'.jpg'
			imgPaths = os.path.join(widerRootDir,'WIDER_'+mode,'images',data['event_list'][eventId], imgName)
			dst = os.path.join(jpegImagesDir,imgName)
			copyfile(imgPaths, dst);

def generateImageSets(widerRootDir, imageSetsDir, mode='train'):
	# check if destination exists
	if not os.path.exists(imageSetsDir):
		os.makedirs(imageSetsDir)

	# read matlab file 
	filepath = os.path.join(widerRootDir, 'wider_face_split')
	filename = os.path.join(filepath,'wider_face_'+mode+'.mat')
	data = sio.loadmat(filename, squeeze_me=True)

	targetPath = os.path.join(imageSetsDir, str(mode)+'.txt')
	fileId = open(targetPath,'w')

	for eventId in range(len(data['file_list'])):
		for file in data['file_list'][eventId]:
			fileId.write(file+'\n')

	fileId.close()


def convertWider2Pascal(SOURCE, TARGET):
	annotationsDir = os.path.join(TARGET,'WIDER','Annotations')
	jpegImagesDir = os.path.join(TARGET,'WIDER','JPEGImages')
	imageSetsDir =  os.path.join(TARGET,'WIDER','ImageSets')

	print 'Generating Annotations: TRAIN'
	generateAnnotations(SOURCE, annotationsDir)
	print 'Generating Annotations: TEST'
	generateAnnotations(SOURCE, annotationsDir, 'val')

	print 'Copying Images : TRAIN'
	copyImages(SOURCE, jpegImagesDir)
	print 'Copying Images : TEST'
	copyImages(SOURCE, jpegImagesDir, 'val')

	print 'Generating ImageSets: TRAIN'
	generateImageSets(SOURCE, imageSetsDir)
	print 'Generating ImageSets: TEST'
	generateImageSets(SOURCE, imageSetsDir, 'val')

    print 'done'


if __name__ == '__main__':
    '''
        Assumes the wider dataset in ./dataset/
        Generates resultant dataset in ./dataset/wider_pascal/
    '''

	ROOT = os.path.dirname(os.path.realpath(__file__))
	WIDER = os.path.join(ROOT, 'dataset/wider')
	TARGET = os.path.join(ROOT,'dataset/wider_pascal')
	convertWider2Pascal(WIDER, TARGET)

	print 'PASCAL VOC format WIDER dataset at',TARGET