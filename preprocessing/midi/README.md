# **MIDI Data Processing**  
This folder provides scripts for converting MIDI files to **MIDI Text Format (MTF)**—the format used by CLaMP 3’s symbolic music encoder (M3)—and, optionally, converting MTF back to MIDI.

## **Scripts**

### **1. [batch_midi2mtf.py](https://github.com/sanderwood/clamp3/blob/main/preprocessing/midi/batch_midi2mtf.py)**  
**Step 1:** Converts MIDI files (`.mid`, `.midi`) to **MTF format** required by **CLaMP 3**.  
- **Execution:**  
  Run the script with the following command:  
  ```bash
  python batch_midi2mtf.py <input_dir> <output_dir> --m3_compatible
  ```
- **Input:** Directory containing MIDI files (`.mid`, `.midi`).  
- **Output:** Directory where **MTF** files (`.mtf`) will be saved.  
- **Important:**  
  - The `--m3_compatible` flag **must be included** to ensure that the output format aligns with CLaMP 3’s symbolic music encoder (M3).  
  - If this flag is omitted, natural language metadata (e.g., title, lyrics) **may be preserved**, which is **not recommended** for use with CLaMP 3.  

### **2. [batch_mtf2midi.py](https://github.com/sanderwood/clamp3/blob/main/preprocessing/midi/batch_mtf2midi.py)** *(Optional)*  
This script converts **MTF files back into MIDI format**.  
- **Execution:**  
  Run the script with the following command:  
  ```bash
  python batch_mtf2midi.py <input_dir> <output_dir>
  ```
- **Input:** Directory containing **MTF files** (or alternatively, MIDI files if reconversion is desired).  
- **Output:** Directory where **MIDI files** will be saved.  