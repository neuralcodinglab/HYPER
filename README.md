# HYPER (brain decoding framework) 🧠 + 🤖 + 📖 = ✨ 

This repo accompanies the original paper ([Dado et al., 2020 (biorxiv)](https://www.biorxiv.org/content/10.1101/2020.07.01.168849v1)), introducing a very powerful yet simple framework for HYperrealistic reconstruction of PERception (HYPER) by elegantly integrating GANs in neural decoding. The goal was to reveal what information was present in the recorded brain responses by reconstructing the original stimuli presented to the participants.

In short, we **visualized** the information in the **brain** while looking at **face images**:

<img src="https://github.com/tdado/HYPER/blob/master/images/hyper.gif" width="620">

## The experiment

Two participants were looking at images of faces while we recorded their brain responses in the MRI scanner. After that, we trained a model to reconstruct what the participants were seeing from their fMRI recordings **alone**. 

Results are ground-breaking:

![](https://github.com/tdado/HYPER/blob/master/images/small.png)

## The trick

The faces in the presented photographs **do not really exist**, but are artificially generated by a progressiveGAN ([PGGAN](https://github.com/tkarras/progressive_growing_of_gans)) from randomly sampled latent vectors. As it turns out, the GAN latent space and the neural manifold have an approximate linear relationship that can be exploited during brain decoding!

🤖🤖🤖


## Required components

This repo contains (demo) code in Jupyter Notebooks to present the approach. All required data to reproduce the results are made available. 

* Python 3.6
* Python libraries: scikit-learn, scipy, numpy, matplotlib, PIL, os, math, pickle, keras, keras-vggface
* [Models (Google Drive)](https://drive.google.com/drive/u/1/folders/1OW0cfnoP8_tZBGWLbpiPPX81QH9pusjv)
* [Preprocessed dataset (Google Drive)](https://drive.google.com/drive/u/1/folders/1xmlusRDS3bTsB78_7RA__RUYyCcAS1jF)
* [PGGAN model for face generation (Github)](https://drive.google.com/drive/folders/15hvzxt_XxuokSmj0uO4xxMTMWVc0cIMU)
* [System requirements for PGGAN (Github)](https://github.com/tkarras/progressive_growing_of_gans)
(incl. [this tfutil library](https://raw.githubusercontent.com/tkarras/progressive_growing_of_gans/master/tfutil.py))
* [5 decision boundaries (Github)](https://github.com/genforce/interfacegan/tree/master/boundaries) disentangled by conditional manipulation

In addition, code regarding preprocessing of the fMRI data (step 1) and creating the final dataset (step 2), folders containing stimuli and reconstructions, and more, can be found on the [Google Drive](https://drive.google.com/drive/u/1/folders/1NEblHtlRFvUyD5CA2sqSVfcGlfJBqw_T).


## A look into the future 🚀

For the first time, we have exploited the GAN latent space for neural decoding during perception of synthetic face photographs. Considering the speed of progress in the field of generative modeling (**new and improved GANs keep on popping up**), the HYPER framework will likely result in even more impressive reconstructions of perception. 

The sky is the limit.


