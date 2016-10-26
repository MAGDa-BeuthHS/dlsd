from inspect import getframeinfo, stack

class Common:
	verbose = False
	def debugInfo(functionName,message):
	    if Common.verbose == True:
	        caller = getframeinfo(stack()[1][0])
	        #os.path.split(caller.filename)[1] # get the name of file
	        print ("%s (%d) : %s"%(functionName,caller.lineno, message))
