# CMS event classification using Deep-Learning Networks

We would like to perform feasibility studies of ML mainstream toolkits with CMS
root based files. We're looking for creation of common framework to explore Big
Data datasets within Machien Learning (ML)/Deep-Learning (DL) frameworks. The
success of this work can lead to adaptation of ML toolkits for High-Energy
Physics (HEP). The problem here is two-fold. On one hand we need to efficiently
handle PB of data and on another we should be able to explore how that amount
of data can be processed via ML DL framework(s). The particular topic of DL
would be to perform event classification of CMS data. Here we can use DL to
either classify events (like trigger) or find new event types via unsupervised
learning.

## Project plan
- collect CMS data for various physics processes, e.g. TTbar, Higgs, Zmumu, etc.
- convert CMS root files into numpy representation using c2numpy [1] utility
  - initially we'll start with charged tracks and extract track parameters,
    e.g. pt, eta, phi, dxy, dz
  - later we can expand parameters to other CMS objects, jets, calo-obejcts, etc.
- get a mix of CMS root files in numpy representation
- build 3D convolution net and perform event classification based on
  know mix of CMS events
- provide benchmark numbers for
  - cost factor of root -> numpy conversion
  - event size representation, root vs numpy, suitable for 3D nets
  - model training
- estimate cost and usability of DL for CMS event classification.

The concrete example of pipeline is available here [2].

## References

[1] https://github.com/diana-hep/c2numpy
[2] https://github.com/diana-hep/c2numpy/tree/master/examples/CMSSW-with-C-interface
