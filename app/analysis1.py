from dlsd.dataset import dataset_generators as dsg
from dlsd.models import SimpleNeuralNetwork as nn
import tensorflow as tf



if __name__ == "__main__":
    # make dataset object containing test/train data objects
    data = dsg.makeData_julyOneWeek2015('datasets/26_8_16_PZS_Belgugung_All_Wide_NanOmited.csv')

    # define parameters and necessary variables
    output_dir = '/Users/ahartens/Desktop/tf'
    max_steps = 10000
    batch_size = 100
    learningRate = 0.3
    n_hidden = 50

    # set up the graph
    graph = tf.Graph()
    with graph.as_default(),tf.device('/cpu:0'):
        # define input/output placeholders
        pl_input = tf.placeholder(tf.float32,shape=[batch_size,data.getNumberInputs()],name="input_placeholder")
        pl_output = tf.placeholder(tf.float32,shape=[batch_size,data.getNumberOutputs()],name="target_placeholder")

        # create neural network and define in graph
        nn = nn.SimpleNeuralNetwork(pl_input,pl_output,n_hidden,learningRate)
        print("outside nueral network")
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
                print(step)
                summary_writer.add_summary(summary_str)
                summary_writer.flush()


