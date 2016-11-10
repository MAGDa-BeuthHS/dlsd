from inspect import getframeinfo, stack
import argparse

class Common:
	verbose = False
	def debugInfo(functionName,message):
	    if Common.verbose == True:
	        caller = getframeinfo(stack()[1][0])
	        #os.path.split(caller.filename)[1] # get the name of file
	        print ("%s (%d) : %s"%(functionName,caller.lineno, message))

	def makeCommandLineArgs():
	    parser = argparse.ArgumentParser(description='Run a neural network')
	    parser.add_argument('-v','--verbose',help='Print error logging messages', required=False)	    
	    parser.add_argument('-r','--restoreSess', help='Restore previous tensorflow session from a file',required=False)
	    parser.add_argument('-m','--makeDataFromSQL',help='Prepare data from SQL output', required=False)
	    parser.add_argument('-tp','--trackPredictions',help='Track progress of the model over training. Prints predictions every 100 steps to outputfile',required=False)
	    parser.add_argument('-t','--train',help='Track progress of the model over training. Prints predictions every 100 steps to outputfile',required=False)

	    args = parser.parse_args()
	    return args

	'''
		
	'''