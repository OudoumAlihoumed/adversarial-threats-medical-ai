"""medattack - adversarial robustness benchmark for tumour detection.

Code for "Adversarial Threats to Safety-Critical Medical AI: A Security
Assessment of 20 Deep Learning Tumour Detectors in Brain MRI and Kidney CT"
by Oudoum Ali Houmed (Gazi University).

Modules
-------
config      experiment settings, dataset definitions and paths
data        image loading, binarisation and the stratified 80/10/10 split
models      the 20 ImageNet backbones and the two-phase training recipe
attacks     FGSM and PGD white-box L-infinity attacks
evaluation  confusion-matrix metrics (accuracy, precision, recall, F1, ...)
similarity  image-quality metrics between clean and adversarial images
plots       confusion matrices, ROC curves and attack example figures
"""

__version__ = "1.0.0"
