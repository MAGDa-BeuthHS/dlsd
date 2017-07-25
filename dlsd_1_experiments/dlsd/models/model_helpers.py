


def _weight_variable(name,shape,stddev):
	dtype = tf.float32
	var = tf.get_variable(name,
			shape,
			tf.truncated_normal_initializer(stddev=stddev,dtype=dtype))
	return var
def _bias_variable(name,shape):
	dtype = tf.float32
	var = tf.get_variable(name,
			shape,
			tf.constant_initializer(0.1))
	return var
def _activation_summary(x)
	tf.histogram_summary(x.op.name+'/activations',x)
	tf.scalar_summary(x.op.name+'/sparsity',tf.nn.zero_fraction(x))

def _conv2d(x,W,name):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME',name=name)

def _max_pool_2x2(x,name):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides = [1,2,2,1], padding='SAME',name=name)
