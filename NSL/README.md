# Nepali Sign Language Character Dataset

This project contains a dataset of Nepali sign language characters. The dataset is organized into two categories based on the background type: `Plain Background` and `Random Background`. Each category contains subfolders representing Nepali characters, with each folder containing a specific number of images.

## Folder Structure
The dataset is structured as follows:

```
├── Plain Background
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── ...
│   └── 35
├── Random Background
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── ...
│   └── 35
```

### Plain Background
- Contains 36 folders labeled from `0` to `35`, each representing a Nepali sign language character.
- Each folder has **1000 images** of the respective character captured against a plain background.

### Random Background
- Contains 36 folders labeled from `0` to `35`, each representing a Nepali sign language character.
- Each folder has **500 images** of the respective character captured against a random background.

## Folder to Nepali Character Mapping
Below is the mapping of folder numbers to Nepali characters:

| Folder Number | Nepali Character |
|---------------|------------------|
| 0             | क               |
| 1             | ख               |
| 2             | ग               |
| 3             | घ               |
| 4             | ङ               |
| 5             | च               |
| 6             | छ               |
| 7             | ज               |
| 8             | झ               |
| 9             | ञ               |
| 10            | ट               |
| 11            | ठ               |
| 12            | ड               |
| 13            | ढ               |
| 14            | ण               |
| 15            | त               |
| 16            | थ               |
| 17            | द               |
| 18            | ध               |
| 19            | न               |
| 20            | प               |
| 21            | फ               |
| 22            | ब               |
| 23            | भ               |
| 24            | म               |
| 25            | य               |
| 26            | र               |
| 27            | ल               |
| 28            | व               |
| 29            | श               |
| 30            | ष               |
| 31            | स               |
| 32            | ह               |
| 33            | क्ष              |
| 34            | त्र              |
| 35            | ज्ञ              |

## Dataset Details
The dataset is designed to support machine learning and deep learning applications, particularly for tasks such as:
- Image classification
- Gesture recognition
- Sign language translation

### Nepali Characters Mapping
Each folder number corresponds to a specific Nepali character in sign language. A mapping table or reference for these numbers should be used in conjunction with this dataset for proper labeling and interpretation.

## Usage
1. Clone the repository or download the dataset.
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the dataset folder and use the images as required for your project.

## Applications
This dataset can be utilized in:
- Training models to recognize Nepali sign language.
- Developing applications for aiding communication with the hearing-impaired community.
- Research in sign language recognition and interpretation.

## Citation
If you use this dataset in your research or projects, please cite this work as:
```
Birat Poudel, Satyam Ghimire, Sijan Bhattarai, Saurav Bhandari. (2025). Nepali Sign Language Character Dataset.
```

## License
This dataset is provided under the [MIT License](LICENSE). Please ensure appropriate use and give credit when using the dataset.

## Acknowledgments
Special thanks to all contributors who helped in capturing, organizing, and validating this dataset, including Satyam Ghimire, Sijan Bhattarai, and Saurav Bhandari.