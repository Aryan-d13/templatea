import os
from marketingspots_template import process_marketingspots_template

# Define input and output directories
input_folder = "OutputEncodedHopefully"
output_folder = "OutputMarketingpotsTesting"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):  # Assuming the files are mp4 videos
        input_path = os.path.join(input_folder, filename)
        
        # Create the output filename
        base_filename, extension = os.path.splitext(filename)
        output_filename = f"{base_filename}_converted{extension}"
        output_path = os.path.join(output_folder, output_filename)
        
        title = "Hey, This text is for Testing purposes, btw love you taylor"

        print(f"Processing {input_path} -> {output_path}")
        
        process_marketingspots_template(
            input_path,
            output_path,
            title
        )

print("Batch processing complete.")
