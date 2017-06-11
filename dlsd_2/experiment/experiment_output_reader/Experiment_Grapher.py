
class Experiment_Grapher:
    def __init__(self):
        self.experiment_output_reader = None
        self.colors = None
        self.target_df = None
        self.predictions_dicts = []        

    def set_experiment_output_directory(self, path):
        self.experiment_directory = path
   
    def extract_data(self):
        reader = Experiment_Output_Reader()
        reader.set_experiment_output_directory(self.experiment_directory)
        reader.extract_data()
        self.target_df = reader.target_df
        self.predictions_dicts = reader.predictions_dicts
    
    def plot_col_idx_from_start_to_end_point(self, idx, s, e):
        self.create_colors_array_of_length(1+len(self.predictions_dicts))
        plt.figure(num=None, figsize=(20, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(self.target_df.iloc[s:e,idx].values, color = self.colors[0])
        plt.title(self.target_df.columns.values[idx])
        for i in range(len(self.predictions_dicts)):
            prediction_df = self.predictions_dicts[i]['df']
            plt.plot(prediction_df.iloc[s:e,idx].values,color=self.colors[i+1])
        self.create_legend()
        plt.savefig('/Users/ahartens/Desktop/sensor_271.png')
    
    def create_colors_array_of_length(self, length):
        self.colors = cm.rainbow(np.linspace(0, 1, length))

    def create_legend(self):
        legend_handles = []
        target_patch = mpatches.Patch(color=self.colors[0],label="Target")
        legend_handles.append(target_patch)
        for i in range(len(self.predictions_dicts)):
            prediction_name = self.predictions_dicts[i]['name']
            patch = mpatches.Patch(color=self.colors[i+1], label=prediction_name)
            legend_handles.append(patch)
        plt.legend(handles=legend_handles)