# nn-ris-optimization
Neural network (NN)-based Reconfigurable Intelligent Surface (RIS) optimization

## Description

This repository contains a main MATLAB script (nn-ris-optimization.m) which reads a given dataset with RIS cascaded channel measurements and optimum phase shift configuration (named measurements_dataset_RIS<RIS_size>_1000.txt, where <RIS_size> corresponds to the size of the RIS under evaluation) and generates a new dataset containing the predictions of the optimum configuration for the RIS using a NN-based approach. An input dataset (measurements_dataset_RIS4_1000.txt) is provided as an example.

The input dataset is a Comma-Separated Values (CSV) file. Each line represents an individual sample with the following fields: The first <RIS_size> elements are the phases of the real RIS-Receiver channel coefficients, the second <RIS_size> elements are the phases of the real Transmitter-RIS channel coefficients, the third <RIS_size> elements are the phases of the sampled RIS-Receiver channel coefficients, the fourth <RIS_size> elements are the phases of the sampled Transmitter-RIS channel coefficients and finally the fifth <RIS_size> elements are the optimum phase shifts of the RIS.

## Copyright

Copyright ⓒ 2024 Pablo Fondo Ferreiro <pfondo@gti.uvigo.es>

This simulator is licensed under the GNU General Public License, version 3 (GPL-3.0). For more information see LICENSE
