# Cantonese Classifier
Fun project on building a Cantonese tone classifier

Just a Cantonese speaker trying to combine signal processing, machine learning and Cantonese, it is by no means perfect

```0_input.py``` allows you to record your own voice into a local folder. In this project, I just used the example of 'si' and its 6 tones.
Usage: 
* ```python 0_input.py record```   record si1..si6 into recordings/
* ```python 0_input.py plot```     analyse recordings/ and my voices, raw and normalised pitch in an exported image (tone_v0.png)
* ```python 0_input.py wave si2``` raw waveform, zoom and spectrogram in an exported image

```tone_classifier.py``` downloads the Cantone dataset, extract features, builds the classifier and evaluate the results
Usage:
* ```python tone_classifier.py download```                   ~1-2 GB into data/cantone/
* ```python tone_classifier.py inspect```                    check labels/speakers were parsed right
* ```python tone_classifier.py train --per-speaker 400```    quick run (~4k clips, a few minutes)
* ```python tone_classifier.py train```                      everything (pitch tracking takes a while, features are cached after the first run)
* ```python tone_classifier.py diagnose```                   per-voice separation of tones 2/5, 1/3, 3/6
* ```python tone_classifier.py train --speakers 'google*' --test-rest``` train/evaluate on Google voices only, then test that model on the other voices
* ```python tone_classifier.py train --speakers 'google*' --test-rest --norm range``` train/evaluate on Google voices only (using a range normalisation method instead of median), then test that model on the other voices
* ```python tone_classifier.py predict recordings/```        test on your own si1_01.wav ... files (Recordings)
