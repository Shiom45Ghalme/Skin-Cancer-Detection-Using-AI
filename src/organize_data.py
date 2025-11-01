import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

class DataOrganizer:
    def __init__(self, raw_data_dir, output_dir):
        """
        Organize HAM10000 dataset into train/val/test folders
        
        Args:
            raw_data_dir: Path to raw downloaded data
            output_dir: Path where organized data will be saved
        """
        self.raw_data_dir = raw_data_dir
        self.output_dir = output_dir
        
    def organize_ham10000(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Organize HAM10000 dataset
        """
        print("Starting data organization...")
        
        # Load metadata
        metadata_path = os.path.join(self.raw_data_dir, 'HAM10000_metadata.csv')
        if not os.path.exists(metadata_path):
            print(f"ERROR: Metadata file not found at {metadata_path}")
            print("Please make sure you extracted the dataset correctly.")
            return
        
        df = pd.read_csv(metadata_path)
        print(f"Found {len(df)} images in metadata")
        
        # Class mapping
        class_names = {
            'nv': 'Melanocytic_nevi',
            'mel': 'Melanoma',
            'bkl': 'Benign_keratosis',
            'bcc': 'Basal_cell_carcinoma',
            'akiec': 'Actinic_keratoses',
            'vasc': 'Vascular_lesions',
            'df': 'Dermatofibroma'
        }
        
        # Print class distribution
        print("\nClass distribution:")
        print(df['dx'].value_counts())
        
        # Create directory structure
        for split in ['train', 'validation', 'test']:
            for class_code, class_name in class_names.items():
                path = os.path.join(self.output_dir, split, class_name)
                os.makedirs(path, exist_ok=True)
        
        # Split data by class to maintain balance
        train_df_list = []
        val_df_list = []
        test_df_list = []
        
        for class_code in class_names.keys():
            class_df = df[df['dx'] == class_code]
            
            # First split: train and temp (val + test)
            train_df, temp_df = train_test_split(
                class_df, 
                test_size=(val_ratio + test_ratio), 
                random_state=42
            )
            
            # Second split: val and test
            val_df, test_df = train_test_split(
                temp_df, 
                test_size=test_ratio/(val_ratio + test_ratio), 
                random_state=42
            )
            
            train_df_list.append(train_df)
            val_df_list.append(val_df)
            test_df_list.append(test_df)
        
        # Combine all classes
        train_df = pd.concat(train_df_list)
        val_df = pd.concat(val_df_list)
        test_df = pd.concat(test_df_list)
        
        print(f"\nSplit sizes:")
        print(f"Training: {len(train_df)} images")
        print(f"Validation: {len(val_df)} images")
        print(f"Test: {len(test_df)} images")
        
        # Copy images to respective folders
        print("\nCopying images...")
        self._copy_images(train_df, 'train', class_names)
        self._copy_images(val_df, 'validation', class_names)
        self._copy_images(test_df, 'test', class_names)
        
        print("\n✓ Data organization complete!")
        print(f"Organized data saved to: {self.output_dir}")
        
    def _copy_images(self, df, split, class_names):
        """
        Copy images to their respective folders
        """
        for idx, row in df.iterrows():
            image_id = row['image_id']
            class_code = row['dx']
            class_name = class_names[class_code]
            
            # Find the image (could be in HAM10000_images_part_1 or part_2)
            src_path = None
            for part in ['HAM10000_images_part_1', 'HAM10000_images_part_2']:
                potential_path = os.path.join(self.raw_data_dir, part, f"{image_id}.jpg")
                if os.path.exists(potential_path):
                    src_path = potential_path
                    break
            
            if src_path is None:
                print(f"Warning: Image {image_id}.jpg not found")
                continue
            
            # Destination path
            dst_path = os.path.join(
                self.output_dir, 
                split, 
                class_name, 
                f"{image_id}.jpg"
            )
            
            # Copy file
            shutil.copy2(src_path, dst_path)
        
        print(f"  ✓ Copied {len(df)} images to {split} folder")

    def verify_organization(self):
        """
        Verify the organized data structure
        """
        print("\n" + "="*50)
        print("DATA ORGANIZATION SUMMARY")
        print("="*50)
        
        for split in ['train', 'validation', 'test']:
            split_path = os.path.join(self.output_dir, split)
            if not os.path.exists(split_path):
                print(f"\n{split.upper()}: NOT FOUND")
                continue
            
            print(f"\n{split.upper()}:")
            classes = os.listdir(split_path)
            total_images = 0
            
            for cls in sorted(classes):
                cls_path = os.path.join(split_path, cls)
                if os.path.isdir(cls_path):
                    count = len([f for f in os.listdir(cls_path) if f.endswith('.jpg')])
                    total_images += count
                    print(f"  {cls}: {count} images")
            
            print(f"  TOTAL: {total_images} images")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    # Configure paths
    raw_data_dir = r"C:\Users\shiom\OneDrive\Desktop\Project xx\data\raw"
    output_dir = r"C:\Users\shiom\OneDrive\Desktop\Project xx\data"
    
    # Create organizer
    organizer = DataOrganizer(raw_data_dir, output_dir)
    
    # Organize the data
    organizer.organize_ham10000(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Verify organization
    organizer.verify_organization()