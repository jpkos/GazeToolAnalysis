# GazeToolAnalysis
For the paper: "Automated Tool Detection with Deep Learning forMonitoring Kinematics and Eye-Hand Coordinationin Microsurgery"
Jani Koskinen, Mastaneh Torkamani-Azar, Ahmed Hussein, Antti Huotarinen, Roman Bednarik
### Data
Data for model training in folder model_train_test_data
  * Test dataset F1-Precision-Recall-Confidence values
Data for case study in folder case_study_data
  * input_data: Tool detection output, eye tracker data, annotated actions and interruptions, and correlation data
  * processed_data: Gaze-tool data after preprocesssing in preprocessing.py

### Scripts
  * preprocessing.py: Data loading and most pre-processing steps
  * analysis.py: Tool and gaze metric calculations
  * correlation_analysis.py: plots for correlations between metrics
  * plots.py: other plots
