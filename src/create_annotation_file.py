import glob
import os

import pandas as pd


# create a class for creating annotation file
class CreateAnnotationFile:
    def __init__(self, dataset_dir_name, split_dataset_name):
        self.dataset_dir_name = dataset_dir_name
        self.split_dataset_name = split_dataset_name

    def create_annotation_df(self):
        """
        Create annotation file for the given subset (train, test, valid)

        Returns:
            annotation_df: pandas DataFrame containing image paths and labels
        """
        image_list = []
        labels = []

        split_dataset_path = os.path.join(self.dataset_dir_name, self.split_dataset_name)

        if not os.path.exists(split_dataset_path):
            raise ValueError(f"Subset path {split_dataset_path} does not exist.")

        for cls in os.listdir(split_dataset_path):            
            images = glob.glob(os.path.join(split_dataset_path, cls, "*.jpg"))
            image_list.extend(images)
            labels.extend([cls] * len(images))

        df = pd.DataFrame({"image_path": image_list, "label": labels})
        self.annotation_df = df
        return df

    def save_annotation_file(self, df: pd.DataFrame, destination_path: str):
        """
        Save the annotation DataFrame to a CSV file in the specified destination path.
        Args:
            df (pd.DataFrame): The annotation DataFrame to save.
            destination_path (str): The path to the destination folder.
        """
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        file_name = f"{self.split_dataset_name}_annotation.csv"
        file_path = os.path.join(destination_path, file_name)
        df.to_csv(file_path, index=False)
        return file_name, file_path


if __name__ == "__main__":
    dataset_dir_name = "breast_cancer_data"
    subset_name = "train"  # can be 'train', 'test', or 'valid'
    destination_path = "test_annotation"

    annotation_creator = CreateAnnotationFile(dataset_dir_name, subset_name)
    annotation_df = annotation_creator.create_annotation_df()
    annotation_creator.save_annotation_file(annotation_df, destination_path)

    print(
        f"Annotation file for {subset_name} subset saved to {destination_path}/{subset_name}_annotation.csv"
    )
