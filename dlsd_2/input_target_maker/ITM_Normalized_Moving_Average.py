from .ITM_Moving_Average import ITM_Moving_Average

class ITM_Normalized_Moving_Average(ITM_Moving_Average):
	def __init__(self):
		super(ITM_Normalized_Moving_Average,self).__init__()
		self.denormalizer_used_in_training = None # if none given, this is set during 

	def make_source_data(self):
		super(ITM_Normalized_Moving_Average,self).make_source_data()
		self.source_dataset_object.denormalizer = self.denormalizer_used_in_training
		self.source_dataset_object.normalize()
		if self.denormalizer_used_in_training is None:
			self.denormalizer_used_in_training = self.source_dataset_object.denormalizer

	def set_denormalizer_with_denormalizer(self, denormalizer):
		self.denormalizer_used_in_training = denormalizer

	def get_denormalizer(self):
		return self.source_dataset_object.denormalizer
	
	def copy_parameters_from_maker(self,mkr):
		super(ITM_Normalized_Moving_Average,self).copy_parameters_from_maker(mkr)
		self.set_denormalizer_with_denormalizer(mkr.get_denormalizer())

	def _common_make(self,maker):
		super(ITM_Normalized_Moving_Average,self)._common_make(maker)
		maker.dataset_object.set_denormalizer(self.source_dataset_object.denormalizer)


	def get_target_df(self):
		return self.denormalizer_used_in_training.denormalize(self.target_maker.dataset_object.df)