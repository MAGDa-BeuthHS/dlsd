import os
import pandas as pd

class Experiment_Output_Reader:
    def __init__(self):
        self.experiment_directory = None
        self.target_df = None
        self.predictions_dicts = []        
        self.path_target = None
        self.path_predictions = None

    def set_experiment_output_directory(self, path):
        self.experiment_directory = path
   
    def extract_data(self):
        self._extract_necessary_file_paths_output_directory()
        self._read_target_data()
        self._read_predictions_data()
        
    def _extract_necessary_file_paths_output_directory(self):
        for root, dirs, files in os.walk(self.experiment_directory, topdown=True):
            for name in files:
                if name == "target.csv":
                    self.path_target = os.path.join(root,name)
            for name in dirs:
                if name == "predictions":
                    self.path_predictions = os.path.join(root,name)
                    
    def _read_target_data(self):
        self.target_df = pd.read_csv(self.path_target,index_col=0)
    
    def _read_predictions_data(self):
        for root, dirs, files in os.walk(self.path_predictions, topdown=True):
            for name in files:
                self._read_prediction_file(root,name)
    
    def _read_prediction_file(self, root, name):
        path = os.path.join(root,name)
        df = pd.read_csv(path,index_col=0)
        the_dict = {'df':df, 'name':name.replace('.csv','')}
        self.predictions_dicts.append(the_dict)