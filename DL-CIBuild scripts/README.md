# To excecute the script:
1. First put them in the same folder as "dataset"
2. Check that all the needed packages like Keras are already installed, otherwise you can refer to "installation instructions" bellow
3. There is a simple example in validationExperiments.py of how to run DL-CIBuild
4. There it is you can start CI build Prediction now!

# Installation instructions to install Theano, TensorFlow and Keras
For Mac users: 

1. Open your main terminal or the Anaconda Prompt and enter the following commands: 

    pip install theano
		
    pip install tensorflow
		
    pip install keras
		
    conda update --all

For Windows and Linux users:

In Spyder, go to Tools and Open Anaconda Prompt. Then enter the following commands:

1. Create a new environment with Anaconda and Python 3.5:
conda create -n tensorflow python=3.5 anaconda

2. Activate the environment: activate tensorflow

3. After this you can install Theano, TensorFlow and Keras:

conda install theano

conda install mingw libpython

pip install tensorflow

pip install keras

4. Update the packages:

conda update --all

5. Run Spyder: spyder
