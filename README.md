<h1 align="center">Transcational Voice AI</h1>

<p align="center">Transcational Voice AI is a library for building voice assistants by combining ASR, Intent recognition and Entity prediction. This library can be used for training models and building predictors to support voice assistants</p>
<p align="center">
  For more info on the Transactional Voice AI deployment codebase, refer <a href="https://github.com/AI4Bharat/transactional-voice-ai_serving">here</a>.
</p>

## Setup
1. Clone the repository
```
git clone https://github.com/AI4Bharat/transactional-voice-ai.git
```
2. Import the conda environment and activate
```
conda env create -n dev-env --file conda-env-setup.yaml
conda activate dev-env
```
3. Clone the [indic-punct](https://github.com/AI4Bharat/indic-punct) library (outside the transactional voice ai folder).
Follow the [installation instructions](https://github.com/AI4Bharat/indic-punct#installation-instructions) to setup indic-punct

## Structure
The project contains four modules as shown in the image. 
<br>
<img src="imgs/structure.jpg" alt="structure" width="500"/>
<br>
Note: Deployment utils provided here are not production ready. For deployment, use [transactional-voice-ai_serving](https://github.com/AI4Bharat/transactional-voice-ai_serving).
