# Gaze Tool Analysis
For the paper: "Automated Tool Detection with Deep Learning forMonitoring Kinematics and Eye-Hand Coordinationin Microsurgery"

Jani Koskinen, Mastaneh Torkamani-Azar, Ahmed Hussein, Antti Huotarinen, Roman Bednarik

University of Eastern Finland, School of Computing

Kuopio University Hospital, Department of Neurosurgery, Institute of Clinical Medicine

### Data
Data for model training in folder model_train_test_data
  * test_results: F1-Precision-Recall-Confidence values for the three experiments
  * training_results: Validation results after each epoch for the three experiments.
Data for case study in folder case_study_data
  * input_data: Tool detection output, eye tracker data, annotated actions and interruptions, and correlation data
  * processed_data: Gaze-tool data after preprocesssing in preprocessing.py
  * output_data: Averaged tool kinematics and gaze-tool distances used in the statistical analyses.

### Scripts
  * preprocessing.py: Data loading and most pre-processing steps
  * analysis.py: Tool and gaze metric calculations
  * correlation_analysis.py: plots for correlations between metrics
  * plots.py: other plots
