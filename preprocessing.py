import os
import cv2
from tqdm import tqdm
from deepface import DeepFace
import numpy as np
import pandas as pd
from pathlib import Path


def crop_and_save_faces(src_folder, dest_folder, output_csv_path, original_csv_path, min_faces=1, max_faces=1):
    """
    Crop faces from images, save them to a new directory, create a CSV file with updated paths, 
    and clean the original CSV file to exclude images where faces could not be detected.
    """
    # Convert to Path objects for better path handling
    src_folder = Path(src_folder)
    dest_folder = Path(dest_folder)
    output_csv_path = Path(output_csv_path)
    original_csv_path = Path(original_csv_path)

    # Create destination folder if it doesn't exist
    dest_folder.mkdir(parents=True, exist_ok=True)

    # Read the original CSV file
    original_df = pd.read_csv(original_csv_path)
    # Strip whitespace from column names
    original_df.columns = original_df.columns.str.strip()
    
    processed_data = []
    skipped_files = []
    
    # Get list of valid image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_files = [f for f in src_folder.iterdir() if f.suffix.lower() in valid_extensions]

    for image_path in tqdm(image_files, desc=f"Processing images in {src_folder}"):
        dest_image_path = dest_folder / image_path.name

        try:
            # Check if image is readable first
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")

            # Extract faces using DeepFace with YOLOv8 backend
            face_objs = DeepFace.extract_faces(
                img_path=str(image_path),
                detector_backend='yolov8',
                enforce_detection=True
            )

            if min_faces <= len(face_objs) <= max_faces:
                # Get the largest face if multiple faces are detected
                face_obj = max(face_objs, key=lambda x: x['face'].shape[0] * x['face'].shape[1])
                face_image = face_obj['face']
                
                # Ensure face image is in correct format
                if face_image.dtype == np.float64 and face_image.max() <= 1.0:
                    face_image = (face_image * 255).astype(np.uint8)
                
                # Convert if not in BGR format
                if len(face_image.shape) == 3 and face_image.shape[2] == 3:
                    bgr_face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                else:
                    bgr_face_image = face_image

                # Save with proper error checking
                success = cv2.imwrite(str(dest_image_path), bgr_face_image)
                if not success:
                    raise ValueError(f"Failed to save image to {dest_image_path}")

                # Get the 'awake' value from the original CSV
                original_row = original_df[original_df['filename'] == image_path.name]
                if not original_row.empty:
                    awake_value = original_row['awake'].iloc[0]
                    processed_data.append([str(dest_image_path), awake_value])
                else:
                    print(f"Warning: No matching entry in original CSV for {image_path.name}")
                    skipped_files.append(image_path.name)

            else:
                skipped_files.append(image_path.name)

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            skipped_files.append(image_path.name)

    # Save processed data to a new CSV
    try:
        processed_df = pd.DataFrame(processed_data, columns=['image_path', 'awake'])
        processed_df.to_csv(output_csv_path, index=False)
        print(f"Processed CSV saved to: {output_csv_path}")
        
        # Print statistics
        print(f"Original images: {len(original_df)}")
        print(f"Processed images: {len(processed_df)}")
        print(f"Skipped images: {len(skipped_files)}")
    
    except Exception as e:
        print(f"Error saving CSV file: {str(e)}")


if __name__ == "__main__":
    # Define source and destination base directories using Path
    base_dir = Path(r'C:\Users\Meiras\Desktop\DL\Driver-drowsiness-detection\driver')
    src_directory = base_dir / 'driver_data'
    dest_directory = base_dir / 'driver_data_cropped'

    # Process train, valid, and test folders separately
    for folder_name in ["train", "valid", "test"]:
        src_folder = src_directory / folder_name
        dest_folder = dest_directory / folder_name
        original_csv_path = src_folder / "_classes.csv"
        output_csv_path = dest_folder / f"{folder_name}_processed.csv"

        if not src_folder.exists():
            print(f"Source folder does not exist: {src_folder}")
            continue

        crop_and_save_faces(
            src_folder=src_folder,
            dest_folder=dest_folder,
            output_csv_path=output_csv_path,
            original_csv_path=original_csv_path,
            min_faces=1,
            max_faces=1
        )