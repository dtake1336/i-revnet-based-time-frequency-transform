# i-RevNet-based time-frequency transform for speech enhancement
In this repository, i-RevNet-based nonlinear time-frequency transform for speech enhancement is impremented using Pytorch.
Our paper can be found [here]() (in preparation).

In our paper, VoiceBank-DEMAND dataset \[2] (available [here](http://dx.doi.org/10.7488/ds/1356)) is used.



### Dependencies
We have tested these codes on follwoing environment:
* Python 3.6.4
* Pytorch 1.4.0
* NumPy 1.17.2
* CUDA Version 10.1
* cuDNN Version 7501


### Usage example
A set of Python codes for training and test are available.
<dl>
<dd> Run "01_train.py" to train a model </dd> 
<dd> Run "02_test.py" to evaluate a model and write .wav files of enhanced speeches </dd> 
</dl>
Note that paths in each code need to be changed for your environment.

### Reference
\[1] D. Takeuchi, K. Yatabe, Y. Koizumi, Y. Oikawa, and N. Harada, “Invertible DNN-based nonlinear time-frequency transform for speech enhancement ,” in 2020 IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP), 2020. (accepted)

\[2] C. Valentini-Botinho, X. Wang, S. Takaki, and J. Yamagishi, “Investigating RNN-based speech enhancement methods for noise-robust Text-to-Speech.,” in 9th ISCA Speech Synth. Workshop, 2016, pp. 146–152.
