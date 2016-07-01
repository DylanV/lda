# lda - Latent Dirichlet Allocation
Implements latent dirichlet allocation with variational inference as in Blei's lda paper

## Build
Standard cmake build. Builds into the /cpp/build/ directory by default

## Launch arguments
* -corpus : path to the document corpus in bag of words style
* -topics : number of topics to train on
* -setting: path to the settings file (optional)
* -param  : folder path to write the model parameters to after training

## Example programs
* Dummy dataset 10 topics
-corpus ../../datasets/dummy2.dat -topics 10 -setting ../default_settings.txt -param ../../param/dummy/
* Associated press dataset 100 topics
-corpus ../../datasets/ap/ap.dat -topics 100 -setting ../default_settings.txt -param ../../param/ap/

## Documentation
Doxygen is supported. To generate run the command : 'doxygen DoxyConfig' in the /cpp/ directory
