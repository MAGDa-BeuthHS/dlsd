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
	    parser.add_argument('-r','--restore', help='Restore from a file',required=False)
	    parser.add_argument('-v','--verbose',help='Print error logging messages', required=False)
	    parser.add_argument('-m','--makeData',help='Prepare data from SQL output', required=False)

	    args = parser.parse_args()
	    return args