import os
import argparse
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP image captioning model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image_path):
    """
    Generate a descriptive caption for the given image using the BLIP model.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

def process_images(input_dir, output_dir):
    """
    Recursively process all images in input_dir, generate captions, 
    and save them in output_dir while maintaining the folder structure.
    """
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    for root, _, files in os.walk(input_dir):
        # Compute the relative path from input_dir
        relative_path = os.path.relpath(root, input_dir)
        output_folder = os.path.join(output_dir, relative_path)

        # Create the corresponding output directory
        os.makedirs(output_folder, exist_ok=True)

        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                input_image_path = os.path.join(root, file)
                caption_text_path = os.path.join(output_folder, f"{os.path.splitext(file)[0]}.txt")

                print(f"Processing: {input_image_path}")

                # Generate caption and save it
                caption = generate_caption(input_image_path)
                with open(caption_text_path, "w", encoding="utf-8") as f:
                    f.write(caption)

                print(f"Caption saved to: {caption_text_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate image captions using BLIP and save them as text files.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing images.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory where captions will be saved.")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)

    print(f"Input Directory: {input_dir}")
    print(f"Output Directory: {output_dir}")

    process_images(input_dir, output_dir)
    print("Processing complete.")

if __name__ == "__main__":
    main()
