# **MIDI Data Processing**
This folder provides scripts for converting MIDI files to **MIDI Text Format (MTF)**—the format used by CLaMP 3’s symbolic music encoder (M3)—and, optionally, converting MTF back to MIDI.

## **Scripts**

### **1. [batch_midi2mtf.py](https://github.com/sanderwood/clamp3/blob/main/preprocessing/midi/batch_midi2mtf.py)**
**Step 1:** Converts MIDI files (`.mid`, `.midi`) to MTF format required by **CLaMP 3**.
- **Setup:**  
  Open `batch_midi2mtf.py` and set the `input_dir` variable to point to your **MIDI directory**.
- **Execution:**  
  Run the script:
  ```bash
  python batch_midi2mtf.py
  ```
- **Input:** MIDI files (`.mid`, `.midi`).
- **Output:** MTF files (`.mtf`).
- **`m3_compatible` Variable:**  
  - **Default:** `True` — This setting removes any natural language content from the MIDI messages to align with CLaMP 3’s training approach.
  - **Optional:** If you wish to preserve text annotations (e.g., title, lyrics), set `m3_compatible = False`.  
    **Note:** This is not recommended for use with CLaMP 3, as the model expects messages without embedded text.

### **2. [batch_mtf2midi.py](https://github.com/sanderwood/clamp3/blob/main/preprocessing/midi/batch_mtf2midi.py)** *(Optional)*  
This script converts MTF files back into MIDI format.
- **Setup:**  
  Open `batch_mtf2midi.py` and set the `input_dir` variable to point to your **MIDI directory**.
- **Execution:**  
  Run the script:
  ```bash
  python batch_mtf2midi.py
  ```
- **Input:** MTF files (or alternatively MIDI files if reconversion is desired).
- **Output:** MIDI files.

> **Important:** Each script requires a manual edit of the `input_dir` variable at the top of the file before running.