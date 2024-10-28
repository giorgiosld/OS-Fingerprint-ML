# OSFingerprintML

A machine learning approach to operating system fingerprinting through analysis of raw memory dumps and pointer graphs.

## Project Overview

This project explores reconstructing information about a target operating system based on raw memory dumps. By leveraging machine learning classifiers on features extracted from pointer graphs and other memory features, we aim to classify and identify OS kernel data structures without prior knowledge of the specific OS.

## Motivation

Precise fingerprinting of an operating system plays a crucial role in applications like penetration testing, intrusion detection, and memory forensics. The ability to identify OS kernel data structures within a memory dump is important for various cybersecurity and forensic applications. This project is part of the T-710-MLCS course in the Cybersecurity Master's Degree at Reykjavik University.

## Objectives

- Explore the classification of memory using machine learning classifiers on features extracted from pointer graphs.
- Investigate the potential of using other memory features for classification, such as statistical properties.
- Experiment with different methods to evaluate their effectiveness empirically.

## Project Contributors
This project was collaboratively developed by [@giorgiosld](https://github.com/giorgiosld) and [@fedemrc](https://github.com/Fedcmm) as a final project for the T-710-MLCS course.

## License
This project is licensed under the MIT License - see the [License](LICENSE) file for details.

## Acknowledgements
- Course: T-710-MLCS (Machine Learning in Cybersecurity), Reykjavik University
- Professor: Hans P. Reiser
- SmartVMI Research Project
