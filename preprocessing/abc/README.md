# ABC Notation Processing

This folder provides scripts, partially adapted from [Wim Vree's ABC tools](https://wim.vree.org/), for converting MusicXML into standard ABC notation, then generating **interleaved ABC** format for **CLaMP 3**, and optionally converting ABC back to MusicXML.

## Scripts

1. **`batch_xml2abc.py`**  
   - **Step 1**: Converts MusicXML files (`.xml`, `.mxl`, `.musicxml`) to standard ABC notation.  
   - **Manual Path Edit**: You must open the script and set `input_dir` to your MusicXML directory.
   - **Output**: Standard ABC files.

2. **`batch_interleaved_abc.py`**  
   - **Step 2**: Converts the **ABC files output from Step 1** into interleaved ABC notation, which is **required by CLaMP 3**.  
   - **Manual Path Edit**: You must open the script and set `input_dir` to your **standard ABC** directory.

3. **`batch_abc2xml.py`** *(Optional)*  
   - Converts ABC notation back to MusicXML, useful for verifying that no information was lost.  
   - **Not required for CLaMP 3**.

## Usage

1. **MusicXML to ABC**:
   ```bash
   python batch_xml2abc.py
   ```
   - Outputs standard ABC files needed for the next step.

2. **ABC to Interleaved ABC**:
   ```bash
   python batch_interleaved_abc.py
   ```
   - Uses the ABC files from Step 1 to produce interleaved ABC format required by CLaMP 3.

> **Note**: Each script requires a manual edit of the `input_dir` variable at the top of the file before running.