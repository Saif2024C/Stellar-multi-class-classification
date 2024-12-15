# Stellar Multi-Class Classification

--- 

# Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
   - [Column Descriptions](#column-descriptions)
   - [Data Source](#data-source)
   - [SDSS DR17 Links](#link-to-access-the-sdss-dr17-data)
3. [Requirements](#requirements)
4. [Project Structure](#project-structure)
5. [Author](#author)


---

## **Project Overview**
This project focuses on building machine learning models to classify astronomical objects into three categories:

- **Galaxy**
- **Star**
- **QSO** (Quasi-Stellar Objects)

The objective is to predict the object class using machine learning techniques like:

1. **SVM** (Support Vector Machines)
2. **XGBoost**
3. **K-Nearest Neighbors (KNN)**
4. **Decision Tree Classification**
5. **Random Tree Classification** 
---

## **Dataset**
The data consists of 100,000 observations of space taken by the SDSS (Sloan Digital Sky Survey). Every observation is described by 17 feature columns and 1 class column which identifies it to be either a star, galaxy or quasar. The table below provides descriptions of all the columns in the dataset. The **`class`** column is the target variable for classification.

| **Column Name**    | **Description**                                                                                      |
|--------------------|------------------------------------------------------------------------------------------------------|
| `obj_ID`           | Object Identifier, the unique value that identifies the object in the image catalog used by the CAS. |
| `alpha`            | Right Ascension angle (at J2000 epoch).                                                             |
| `delta`            | Declination angle (at J2000 epoch).                                                                 |
| `u`                | Ultraviolet filter in the photometric system.                                                       |
| `g`                | Green filter in the photometric system.                                                             |
| `r`                | Red filter in the photometric system.                                                               |
| `i`                | Near Infrared filter in the photometric system.                                                     |
| `z`                | Infrared filter in the photometric system.                                                          |
| `run_ID`           | Run Number used to identify the specific scan.                                                      |
| `rereun_ID`        | Rerun Number to specify how the image was processed.                                                |
| `cam_col`          | Camera column to identify the scanline within the run.                                              |
| `field_ID`         | Field number to identify each field.                                                                |
| `spec_obj_ID`      | Unique ID used for optical spectroscopic objects (observations with the same ID share the same class).|
| `class`            | **Target column**: Object class (Galaxy, Star, or Quasar).                                          |
| `redshift`         | Redshift value based on the increase in wavelength.                                                 |
| `plate`            | Plate ID, identifies each plate in SDSS.                                                            |
| `MJD`              | Modified Julian Date, used to indicate when the SDSS data was collected.                            |
| `fiber_ID`         | Fiber ID that identifies the fiber pointing light at the focal plane in each observation.           |

**Data Source**: `stellar-classification-dataset-sdss17.zip`

**Link to Access the SDSS DR17 Data**
For those interested in exploring the SDSS data, you can access it here:

**Sloan Digital Sky Survey DR17**: https://www.sdss.org/dr17/ , 
**SDSS SkyServer (Data Query Tool)**: https://skyserver.sdss.org/dr17


---

## **Requirements**
To run this project, you need the following Python libraries:

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning tools
- **xgboost**: XGBoost model
- **matplotlib** and **seaborn**: Data visualization

Install dependencies using:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## **Project Structure**
The repository is organized as follows:

```plaintext
ML_astro/
├── star_classification.csv       # Dataset file
├── ML_astro_refined.ipynb        # Main script for training and evaluation
├── README.md                     # Project documentation
```

## **Author** 
This project was developed for academic purposes. Contributions and suggestions are welcome.

