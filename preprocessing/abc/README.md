# **ABC Notation Processing**
This folder provides scripts—partially adapted from [Wim Vree's ABC tools](https://wim.vree.org/svgParse/index.html)—for converting MusicXML files into **standard ABC notation**, then generating **interleaved ABC** format required by **CLaMP 3**, and optionally converting ABC back to MusicXML.

## **Scripts**

### **1. [batch_xml2abc.py](https://github.com/sanderwood/clamp3/blob/main/preprocessing/abc/batch_xml2abc.py)**
**Step 1:** Converts MusicXML files (`.xml`, `.mxl`, `.musicxml`) to **standard ABC notation**.
- **Setup:**  
  Open `batch_xml2abc.py` and set the `input_dir` variable to point to your **MusicXML directory**.
- **Execution:**  
  Run the script:
  ```bash
  python batch_xml2abc.py
  ```
- **Input:** MusicXML files (`.mxl`, `.xml`, `.musicxml`).
- **Output:** Standard ABC files (`.abc`).

### **2. [batch_interleaved_abc.py](https://github.com/sanderwood/clamp3/blob/main/preprocessing/abc/batch_interleaved_abc.py)**
**Step 2:** Converts the **ABC files output from Step 1** into **interleaved ABC notation** required by CLaMP 3.
- **Setup:**  
  Open `batch_interleaved_abc.py` and set the `input_dir` variable to point to your **Standard ABC directory**.
- **Execution:**  
  Run the script:
  ```bash
  python batch_interleaved_abc.py
  ```
- **Input:** Standard ABC files (`.abc`).
- **Output:** Interleaved ABC files (`.abc`).

### **3. [batch_abc2xml.py](https://github.com/sanderwood/clamp3/blob/main/preprocessing/abc/batch_abc2xml.py)** *(Optional)*  
This script converts **Interleaved ABC files** back into **MusicXML format**.  
- **Note:** This step is **not required for CLaMP 3**.
- **Setup:**  
  Open `batch_abc2xml.py` and set the `input_dir` variable to point to your **Interleaved ABC directory**.
- **Execution:**  
  Run the script:
  ```bash
  python batch_abc2xml.py
  ```
- **Input:** Interleaved ABC files (`.abc`).
- **Output:** MusicXML files (`.xml`).

> **Important:** Each script requires a manual edit of the `input_dir` variable at the top of the file before running.