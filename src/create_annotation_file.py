import glob
import pandas as pd
import os

# create a class for creating annotation file
class CreateAnnotationFile:
    def __init__(self, dataset_dir_name, subset_name):
        self.dataset_dir_name = dataset_dir_name
        self.subset_name = subset_name

    def create_annotation(self):
        image_list = []
        labels = []

        subset_path = os.path.join(self.dataset_dir_name, self.subset_name)
        if not os.path.exists(subset_path):
            raise ValueError(f"Subset path {subset_path} does not exist.")

        for path in os.listdir(subset_path):
            images = glob.glob(f'{subset_path}/{path}/*jpg')
            image_list.extend(images)
            labels.extend([path] * len(images))

        data = {'image_path': image_list, 'label': labels}
        self.annotation_df = pd.DataFrame(data)

    def save_annotation_file(self, destination_path):
        # create annotation folder if not exists
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)

        # save annotation file in current directory
        self.annotation_df.to_csv(f'{destination_path}/{self.subset_name}_annotations.csv', index=False)


if __name__ == "__main__":
    # create annotation folder if not exists
    if not os.path.exists('annotation'):
        os.makedirs('annotation')

    # create annotation file in annotation folder
    train_annotation = CreateAnnotationFile(
        dataset_dir_name='breast_cancer_data', subset_name='train')
    train_annotation.create_annotation()
    train_annotation.save_annotation_file('annotation')

    test_annotation = CreateAnnotationFile(
        dataset_dir_name='breast_cancer_data', subset_name='test')
    test_annotation.create_annotation()
    test_annotation.save_annotation_file('annotation')
    
    val_annotation = CreateAnnotationFile(
        dataset_dir_name='breast_cancer_data', subset_name='valid')
    val_annotation.create_annotation()
    val_annotation.save_annotation_file('annotation')
