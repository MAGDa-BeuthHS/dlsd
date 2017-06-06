from dlsd_2.experiment.Experiment import Experiment
import logging

logging.basicConfig(level=logging.DEBUG)#filename='17_05_04_dlsd_2_trials.log',)

def main():
	exp = Experiment()
	exp.set_experiment_root_path('/Users/ahartens/Desktop/Work/dlsd_2_trials/trial_1')
	exp.run_experiment()

if __name__=="__main__":
	main()



