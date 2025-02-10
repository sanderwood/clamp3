# MIDI Data Processing

This folder provides scripts for converting MIDI files to **MIDI Text Format (MTF)**—the format used by CLaMP 3’s symbolic music encoder (M3)—and optionally converting MTF back to MIDI.

## Scripts

1. **`batch_midi2mtf.py`**  
   - **Step 1**: Converts MIDI files (`.mid`, `.midi`) to MTF format required by **CLaMP 3**.  
   - **`m3_compatible` Variable**:  
     - Default is `True`, which removes any natural language content within the MIDI messages for alignment with CLaMP 3’s training approach.  
     - If you want to preserve text annotations (title, lyrics), set `m3_compatible = False` in the script. However, **this is not recommended for use with CLaMP 3**, as the model expects messages without embedded text.  
   - **Manual Path Edit**: Open the script and set `input_dir` to your MIDI directory.  
   - **Output**: MTF files.

2. **`batch_mtf2midi.py`** *(Optional)*  
   - Converts MTF files back into MIDI format.  
   - Useful for verifying no information was lost in the conversion.  
   - **Not required for CLaMP 3**.

## Usage

1. **MIDI to MTF** (mandatory for CLaMP 3):
   ```bash
   python batch_midi2mtf.py
   ```
   - Creates MTF files for CLaMP 3.

2. **MTF to MIDI** (optional):
   ```bash
   python batch_mtf2midi.py
   ```
   - Converts MTF files back to MIDI if needed for verification.

> **Note**: Each script requires a manual edit of the `input_dir` variable at the top of the file before running.