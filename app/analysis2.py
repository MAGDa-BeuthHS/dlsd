from dlsd.dataset import dataset_generators as dsg
from dlsd.models import SimpleNeuralNetwork as nn
from dlsd import Common as c
import tensorflow as tf



if __name__ == "__main__":
    # set debugInfo verbose variable
    c.verbose = True
    
    # create a FullDataSet object containing train/test data as well as next_batch() method
    #data = dsg.makeData_allSensorsInAllOutWithTimeOffset('/Users/ahartens/Desktop/Temporary/24_10_16_wideTimeSeriesBelegung_naDropped.csv',False)
    data = dsg.makeData_allSensorsInOneOutWithTimeOffset('/Users/ahartens/Desktop/Temporary/24_10_16_wideTimeSeriesBelegung_naDropped.csv',False)
    
    # remake data from SQL output
    '''
    data = dsg.makeData_allSensorsInAllOutWithTimeOffset(inputFilePath = '/Users/ahartens/Desktop/Work/24_10_16_PZS_Belegung_limited.csv',
        remakeData =True,
        outputFilePath ='/Users/ahartens/Desktop/Temporary/16_10_26_wideTimeSeriesBelegung.csv',
        saveOutputFile = True)
    '''
    # define parameters and necessary variables
    output_dir = '/Users/ahartens/Desktop/tf'
    max_steps = 1000
    batch_size = 100
    learningRate = 0.3
    n_hidden = 40

    # set up the graph
    graph = tf.Graph()
    with graph.as_default(),tf.device('/cpu:0'):
        # define input/output placeholders
        pl_input = tf.placeholder(tf.float32,shape=[batch_size,data.getNumberInputs()],name="input_placeholder")
        pl_output = tf.placeholder(tf.float32,shape=[batch_size,data.getNumberOutputs()],name="target_placeholder")

        # create neural network and define in graph
        c.debugInfo(__name__,"Creating neural network")
        nn = nn.SimpleNeuralNetwork(pl_input,pl_output,n_hidden,learningRate)
        
        summary_op = tf.merge_all_summaries()


    with tf.Session(graph = graph) as sess:
        summary_writer = tf.train.SummaryWriter(output_dir, sess.graph)
        sess.run(tf.initialize_all_variables())

        for step in range(max_steps):
            myFeedDict = dsg.fill_feed_dict(data.train,
                                       pl_input,
                                       pl_output,
                                       batch_size)
            loss_value,summary_str = sess.run([nn.optimize,summary_op],feed_dict = myFeedDict)
            if(step%100 == 0):
                summary_writer.add_summary(summary_str)
                summary_writer.flush()
                
                myFeedDict = dsg.fill_feed_dict(data.test,
                                       pl_input,
                                       pl_output,
                                       batch_size)
                testMeanErrorValue = sess.run(nn.evaluation,feed_dict = myFeedDict)

                c.debugInfo(__name__,"Training step : %d of %d"%(step,max_steps))


'''
     with tf.Session(graph = graph) as sess:
        summary_writer = tf.train.SummaryWriter(output_dir, sess.graph)
        sess.run(tf.initialize_all_variables())

        for step in range(max_steps):
            myFeedDict = dsg.fill_feed_dict(data.train,
                                       pl_input,
                                       pl_output,
                                       batch_size)
            loss_value,trainMeanError,summary_str = sess.run([nn.optimize,nn.error,summary_op],feed_dict = myFeedDict)
            if(step%100 == 0):
                summary_writer.add_summary(summary_str)
                summary_writer.flush()
                myFeedDict = dsg.fill_feed_dict(data.test,
                                       pl_input,
                                       pl_output,
                                       batch_size)
                testMeanErrorValue = sess.run(nn.evaluation,feed_dict = myFeedDict)
                c.debugInfo(__name__,"Training step : %d of %d  testing error:"%(step,max_steps))
'''
