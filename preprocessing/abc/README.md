# **ABC Notation Processing**  
This folder provides scripts—partially adapted from [Wim Vree's ABC tools](https://wim.vree.org/svgParse/index.html)—for converting MusicXML files into **standard ABC notation**, then generating **interleaved ABC** format required by **CLaMP 3**, and optionally converting ABC back to MusicXML.

## **Scripts**

### **1. [batch_xml2abc.py](https://github.com/sanderwood/clamp3/blob/main/preprocessing/abc/batch_xml2abc.py)**  
**Step 1:** Converts MusicXML files (`.xml`, `.mxl`, `.musicxml`) to **standard ABC notation**.  
- **Execution:**  
  Run the script with the following command:  
  ```bash
  python batch_xml2abc.py <input_dir> <output_dir>
  ```
- **Input:** Directory containing MusicXML files (`.mxl`, `.xml`, `.musicxml`).  
- **Output:** Directory where converted **Standard ABC** (`.abc`) files will be saved.  

### **2. [batch_interleaved_abc.py](https://github.com/sanderwood/clamp3/blob/main/preprocessing/abc/batch_interleaved_abc.py)**  
**Step 2:** Converts the **ABC files output from Step 1** into **interleaved ABC notation** required by CLaMP 3.  
- **Execution:**  
  Run the script with the following command:  
  ```bash
  python batch_interleaved_abc.py <input_dir> <output_dir>
  ```
- **Input:** Directory containing **Standard ABC** files (`.abc`).  
- **Output:** Directory where **Interleaved ABC** files (`.abc`) will be saved *(required for CLaMP 3)*.  

### **3. [batch_abc2xml.py](https://github.com/sanderwood/clamp3/blob/main/preprocessing/abc/batch_abc2xml.py)** *(Optional)*  
This script converts **Interleaved ABC files** back into **MusicXML format**.  
- **Note:** This step is **not required for CLaMP 3**.  
- **Execution:**  
  Run the script with the following command:  
  ```bash
  python batch_abc2xml.py <input_dir> <output_dir>
  ```
- **Input:** Directory containing **Interleaved ABC** files (`.abc`).  
- **Output:** Directory where **MusicXML** files (`.xml`) will be saved.  