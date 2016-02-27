RapidMiner Extension - DL4J Extension Pack
=============================
Version: 0.6.0-beta 
------------------------------------

This is an under-develop extension of RapidMiner supported by RapidMiner China.
This plug-in integrates RaoidMiner with deep learning features by using the third party library deeplearning4j aka. DL4J,
where DL4J is an java-based, open source deeplearning lib developed by Skymind. 

### Prerequisite
* The extension has all it's libraries wrapped inside, thus the only prerequisite to use it is to have RapidMiner 7.0 or above installed.

### Features

The current version contains operators implementing thr0ee kinds of neural network models, namely:

Multi-layered Neural Network: 
This is the general type of neural network.
Apply to general labeled numerical example set and could be used for general purpose.

Convultional Neural Network (CNN):
Particularly developed for image process but also works on certain general tasks.
Apply to converted labeled image sources, as well as certain general labeled numerical example set.

Word2Vec: 
For text process only. 
Works on .txt files which contains raw sentences, currently, only English is supported.