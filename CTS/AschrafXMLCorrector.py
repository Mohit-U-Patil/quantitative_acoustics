# -*- coding: utf-8 -*-
# 
VERSION = 1.1
#	This script corrects the raw data (XML format) collected during the reliability part of the cortical thickness study of Aschraf Danun
#	1) Sensor positions were not input, and should be within [0.04, 0.39] in 0.01 steps.
#	2) time was not given, should be set to the measurement day
#	3) first and last measurement should be deleted, so remaining measurements are [0.05, 0.39]
#	CHANGELOG:
#	1.1 - phantom field is now corrected to phantom id: 119
#		
#	Usage: 
#		define subclass, implement the correct() method, then create an object and call the correctAllFiles() method
#		prepended comment string is defined by self.commentString, which can be overloaded as well

import pandas as pd
import glob
import xml.etree.ElementTree as ET
import os.path
from datetime import datetime, timedelta
import logging
import re
from abc import ABCMeta, abstractmethod
from pBSMDXMLCorrector import pBSMDXMLCorrector

class AschrafXMLCorrector(pBSMDXMLCorrector):
	def correct(self, inFile, outFile):
		tree = ET.parse(inFile)	#FileNotFoundError when file is not found
		root = tree.getroot()
		
		#get measurement numbers within that file
		measNrs = [meas.get("nr") for meas in root.findall('measurement')]
		firstMeas = root.find("./measurement[@nr='{}']".format(measNrs.pop(0)))
		root.remove(firstMeas)
		lastMeas = root.find("./measurement[@nr='{}']".format(measNrs.pop()))
		root.remove(lastMeas)
		
		#set the remaining sensor positions to [0.05, 0.38]
		assert len(measNrs) == 34, "File {} did contain {} instead of the expected 34 measurements after deletion of the first and second measurement".format(os.path.basename(inFile), len(measNrs))
		i = 0.05
		for nr in measNrs:
			oldEle = root.find("./measurement[@nr='{}']/signal/channel".format(nr))
			oldEle.set('position', str(i))
			i += 0.01
		
		# set the datetime to the date the measurement was taken
		newDate = datetime(2016, 10, 15)
		oldEle = root.find("./meta/timestamp")
		dt = newDate - self.DATEREF 		# gives timedelta
		oldEle.text = str(int(dt.total_seconds()))
		
		#set the phantom to its id:119
		oldEle = root.find("./meta/phantom")
		oldEle.text = "119"
		
		tree.write(outFile)
		logging.info("Wrote file {}".format(outFile))
		
if __name__ == "__main__":
	logging.basicConfig(filename='xmlCorrector.log',level=logging.INFO)
	xc = AschrafXMLCorrector("./wrongMeas", "./corrMeas")
	xc.correctAllFiles()