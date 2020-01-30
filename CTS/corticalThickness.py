import glob
import os
from bsmd.utilities import make_sure_path_exists
from bsmdMeasurement import bsmdMeasurement
import bsmdAlgorithm
import matplotlib.pylab as plt

OUTPUTDIR	= r'.\porosityOutput'
DATADIR		= r'T:\measurements-svn\cortical_thickness-ashraf\Phantom_radial_1.5-7.5mm'

#make sure outdirectory and logfile exists
make_sure_path_exists(OUTPUTDIR)

TIMEFFT			= 20000
SPACEFFT		= 350

algo = bsmdAlgorithm.PhaseVelocity_2DFFTGauss()

for root, dirs, files in os.walk(DATADIR):
	# get all files satisfying the filter
	for file in glob.glob(os.path.join(root,"*.xml")):
			print(file)
		#if os.path.basename(file) == "point2_r25mm.xml":
			meas = bsmdMeasurement(xFFTNr = TIMEFFT, yFFTNr = SPACEFFT)
			meas.readStepPhantomXMLData(file)
			meas.plot2D(meas.dataFFT, showImages = True, xRange = (2800, 3200), yRange = (4,11))
			try:
				c = algo.phaseVelocity(meas, freq = (2900, 3100), wavenumber = (5,7), debug_plot = False, sumOfSquares_threshold = 0.003)
			except:
				c = algo.phaseVelocity(meas, freq = (2900, 3100), wavenumber = (4,6), debug_plot = False, sumOfSquares_threshold = 0.01)
			c.T.plot()
			plt.title(os.path.basename(file))
			plt.show()
			plt.close()
		
		