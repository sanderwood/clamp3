# **Data Processing Overview**
The [preprocessing/](https://github.com/sanderwood/clamp3/tree/main/preprocessing) folder contains scripts and utilities for converting and processing musical data in various formats. These scripts are organized into three primary categories—**ABC Notation**, **Audio**, and **MIDI**—with each category housed in its own subfolder.

## **Folder Structure**

### **1. [abc/](https://github.com/sanderwood/clamp3/tree/main/preprocessing/abc)**
This folder contains scripts for processing **ABC notation** files. It includes utilities for converting to and from MusicXML, making it easier to work with sheet music representations and standardize data across different formats.

### **2. [audio/](https://github.com/sanderwood/clamp3/tree/main/preprocessing/audio)**
This folder is dedicated to audio processing. It provides scripts for extracting features from audio files using **MERT**. These tools convert raw audio into feature representations required by the system, facilitating robust audio analysis.

### **3. [midi/](https://github.com/sanderwood/clamp3/tree/main/preprocessing/midi)**
This folder contains scripts for processing **MIDI** data. It includes converters to transform MIDI files into other formats, such as **MTF (MIDI Text Format)**, to support downstream processing and analysis of performance signals.

> **Note**: For more detailed usage instructions and configuration options, please refer to the individual README files located in each subfolder.