from __future__ import print_function
from __future__ import absolute_import
__author__ = 'tylin'
__version__ = '1.0.1'

import json
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString

import json
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
# from skimage.draw import polygon
import urllib
import copy
import itertools
#from . import mask
import os
try:
    unicode        # Python 2
except NameError:
    unicode = str  # Python 3

class COCO:
    def __init__(self, annotation_file=None):
        print('found json file')
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset = {}
        self.anns = []
        self.imgToAnns = {}
        self.catToImgs = {}
        self.imgs = {}
        self.cats = {}
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            print('Done (t=%0.2fs)'%(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        anns = {}
        imgToAnns = {}
        catToImgs = {}
        cats = {}
        imgs = {}
        if 'annotations' in self.dataset:
            imgToAnns = {ann['image_id']: [] for ann in self.dataset['annotations']}
            anns =      {ann['id']:       [] for ann in self.dataset['annotations']} #THIS IS ANNO ID! ITS DIFFERENT FROM THE IMAGE ID IN SELF.DATASET['IMAGES']. IN DATASET['IMAGES'], THE ID IS THE NAME OF THE IMAGE
            for ann in self.dataset['annotations']:
                imgToAnns[ann['image_id']] += [ann] #CREATED DICTIONARY MAPPING IMAGE ID TO THE ANNOTATION (IMAGEID, ID, CATEGORYID, BBOX COORDINATES)
                anns[ann['id']] = ann #CREATED MAPPING FROM ANNOTATION ID TO ANNOTATION

        if 'images' in self.dataset:
            imgs      = {im['id']: {} for im in self.dataset['images']}
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            cats = {cat['id']: [] for cat in self.dataset['categories']}
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat['name']
            catToImgs = {cat['id']: [] for cat in self.dataset['categories']}
            if 'annotations' in self.dataset:
                for ann in self.dataset['annotations']:
                    catToImgs[ann['category_id']] += [ann['image_id']] #MAPPING FROM CATEGORY ID TO ALL IMAGES THAT CONTAIN THAT CATEGORY ANIMAL

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('%s: %s'%(key, value))

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[]):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               # not applicable areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               # Not applicable: iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        #IF THERE ARE NO CONDITIONS PASSED, RETURN ALL ANNOTATION IDS IN THE FILE
        if len(imgIds) == len(catIds) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                # this can be changed by defaultdict
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]
            #anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]
        #if not iscrowd == None:
        #    ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        #else:
        ids = [ann['id'] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if type(catNms) == list else [catNms]
        supNms = supNms if type(supNms) == list else [supNms]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        '''
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        catIds = catIds if type(catIds) == list else [catIds]

        if len(imgIds) == len(catIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
        return list(ids)

    def loadAnns(self, ids):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer/string ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if type(ids) == list:
            return [self.anns[id] for id in ids]
        elif type(ids) == int or type(ids)== str:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if type(ids) == list:
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer/string ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if type(ids) == list:
            return [self.imgs[id] for id in ids]
        #elif type(ids) == int:
        return [self.imgs[ids]]


coco = COCO('/home/paperspace/Desktop/faster-rcnn.pytorch/trainval.json')
tmp = {}
for k in coco.imgs.keys():
    tmp = coco.imgs[k]
    tmp["object"] = coco.imgToAnns[k]
    # appended all annotations belonging to an image
    xml = dicttoxml(tmp, attr_type=False, custom_root = 'annotation')
    dom1 = parseString(xml)
    myfile = open('/home/paperspace/Desktop/faster-rcnn.pytorch/data/cct_devkit/CCT20/Annotations/' + k +'.xml', "w")
    myfile.write(dom1.toprettyxml())
    '''
    file2 = open('/home/prachi/faster-rcnn.pytorch/data/cct_devkit/CCT20/ImageSets/testtrans.txt', "a")
    file2.write(k+'\n')
    '''
