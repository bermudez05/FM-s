# Foundation Models in Medical Imaging: CheXzero & MI-Zero

> Exploring how **vision–language foundation models** perform zero-shot pathology detection in medical images. The project reproduces and evaluates **CheXzero** on chest X-rays and studies **MI-Zero** for cancer detection in histopathology.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![CLIP](https://img.shields.io/badge/OpenAI-CLIP-412991)

Academic project for *Mathematics for Machine Learning*, Universidad Nacional de Colombia. This repository extends the original [CheXzero](https://github.com/rajpurkarlab/CheXzero) codebase by Rajpurkar Lab (Stanford).

## About this project

Foundation models are trained once on large, mostly unlabelled data and then adapted to many tasks. CheXzero is a CLIP-style model trained on chest X-rays paired with free-text radiology reports. It can detect pathologies **without any explicit labels**, by comparing image embeddings with text prompts such as "pneumonia" / "no pneumonia".

We studied:

- **CheXzero (chest X-rays):** we ran the full zero-shot pipeline (preprocessing, inference, evaluation) and analysed its generalisation, accuracy and efficiency in the [`notebooks/zero_shot.ipynb`](notebooks/zero_shot.ipynb) experiment notebook.
- **MI-Zero (histopathology):** we studied the model and attempted to run it for cancer detection in whole-slide images. The implementation could not be completed because of operating-system and library incompatibilities on our machines; the paper and video document what we learned.

## Deliverables

- 📄 **Paper:** [Foundational_Models_MAM.pdf](entregables/Foundational_Models_MAM.pdf)
- 📊 **Slides:** [Foundation models in medical imaging: MI-Zero and CheXzero](entregables/)
- 🎥 **Video:** [YouTube](https://youtu.be/s-0Sw0E_N_M?si=4xYvLWEe4DU6l9oR)

## Related repositories

Solutions to the *Understanding Deep Learning* (Prince, 2023) notebooks, by each team member:

- Nicolás Álvarez Triana: [Notebooks-Understanding-DL](https://github.com/niclasDc/Notebooks-Understanding-DL)
- **Laura Nicole Bermúdez Santa**: [Understanding-DL---MPAM](https://github.com/bermudez05/Understanding-DL---MPAM)
- Juan Sebastián Castro Pardo: [MathsForML](https://github.com/SebastianCastro23/MathsForML)

---

<sub>Everything below is the original CheXzero documentation, kept for reference.</sub>

2. **Matching radiologist performance on different tasks on an external test set:** Our method performed on par with human performance when evaluated on an external validation set (CheXpert) of chest x-ray images labeled for the presence of 14 different conditions by multiple radiologists.
3. **Outperforming approaches that train on explicitly labeled data on an external test set:** Using no labels, we outperformed a fully supervised approach (100% of labels) on 3 out of the 8 selected pathologies on a dataset (PadChest) collected in a different country. We further demonstrated high performance (AUC > 0.9) on 14 findings and at least 0.700 on 53 findings out of 107 radiographic findings that the method had not seen during training.
</details>

## Dependencies

To clone all files:

`git clone https://github.com/rajpurkarlab/CheXzero.git`

To install Python dependencies:

`pip install -r requirements.txt`

## Data

### Training Dataset

1. Download images come from [MIMIC-CXR JPG] https://physionet.org/content/mimic-cxr-jpg/2.0.0/ and reports from [MIMIC-CXR Database](https://physionet.org/content/mimic-cxr/2.0.0/) Note: in order to gain access to the data, you must be a credentialed user as defined on [PhysioNet](https://physionet.org/settings/credentialing/).
2. Copy the dataset into the `data/` directory.
3. Run `python run_preprocess.py`
4. This should preprocess the chest x-ray images into a Hierarchical Data Format (HDF) format used for training stored at `data/cxr.h5` and extract the impressions section as text from the corresponding chest x-ray radiology report stored at `data/mimic_impressions.csv` .

### Evaluation Dataset

#### CheXpert Dataset

The CheXpert dataset consists of chest radiographic examinations from Stanford Hospital, performed between October 2002
and July 2017 in both inpatient and outpatient centers. Population-level characteristics are unavailable for the CheXpert test
dataset, as they are used for official evaluation on the CheXpert leaderboard.

The main data (CheXpert data) supporting the results of this study are available at https://aimi.stanford.edu/chexpert-chest-x-rays.

The CheXpert **test** dataset has recently been made public, and can be found by following the steps in the [cheXpert-test-set-labels](https://github.com/rajpurkarlab/cheXpert-test-set-labels) repository.

#### PadChest Dataset

The PadChest dataset contains chest X-rays that were interpreted by 18 radiologists at the Hospital Universitario de San Juan,
Alicante, Spain, from January 2009 to December 2017. The dataset contains 109,931 image studies and 168,861 images.
PadChest also contains 206,222 study reports.

The [PadChest](https://arxiv.org/abs/1901.07441) is publicly available at https://bimcv.cipf.es/bimcv-projects/padchest. Those who would like to use PadChest for experimentation should request access to PadChest at the [link](https://bimcv.cipf.es/bimcv-projects/padchest).

### Model Checkpoints

Model checkpoints of CheXzero pre-trained on MIMIC-CXR are publicly available at the following [link](https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno?usp=sharing). Download files and save them in the `./checkpoints/chexzero_weights` directory.

## Running Training

Run the following command to perform CheXzero pretraining.

```bash
python run_train.py --cxr_filepath "./data/cxr.h5" --txt_filepath "data/mimic_impressions.csv"
```

### Arguments

- `--cxr_filepath` Directory to load chest x-ray image data from.
- `--txt_filepath` Directory to load radiology report impressions text from.

Use `-h` flag to see all optional arguments.

## Zero-Shot Inference

See the following [notebook](https://github.com/rajpurkarlab/CheXzero/blob/main/notebooks/zero_shot.ipynb) for an example of how to use CheXzero to perform zero-shot inference on a chest x-ray dataset. The example shows how to output predictions from the model ensemble and evaluate performance of the model if ground truth labels are available.

```python
import zero_shot

# computes predictions for a set of images stored as a np array of probabilities for each pathology
predictions, y_pred_avg = zero_shot.ensemble_models(
    model_paths=model_paths,
    cxr_filepath=cxr_filepath,
    cxr_labels=cxr_labels,
    cxr_pair_template=cxr_pair_template,
    cache_dir=cache_dir,
)
```

### Arguments

- `model_paths: List[str]`: List of paths to all checkpoints to be used in the ensemble. To run on a single model, input a list containing a single path.
- `cxr_filepath: str`: Path to images `.h5` file
- `cxr_labels: List[str]`: List of pathologies to query in each image
- `cxr_pair_templates: Tuple[str, str]`: constrasting templates used to query model (see Figure 1 in article for visual explanation).
- `cache_dir: str`: Directory to cache predictions of each checkpoint, use to avoid recomputing predictions.

In order to use CheXzero for zero-shot inference, ensure the following requirements are met:

- All input _`images`_ must be stored in a single `.h5` (Hierarchical Data Format). See the [`img_to_h5`](https://github.com/rajpurkarlab/CheXzero/blob/main/preprocess_padchest.py#L156) function in [preprocess_padchest.py](https://github.com/rajpurkarlab/internal-chexzero/blob/cleanversion/preprocess_padchest.py) for an example of how to convert a list of paths to `.png` files into a valid `.h5` file.
- The _ground truth `labels`_ must be in a `.csv` dataframe where rows represent each image sample, and each column represents the binary labels for a particular pathology on each sample.
- Ensure all [model checkpoints](https://drive.google.com/drive/folders/1makFLiEMbSleYltaRxw81aBhEDMpVwno?usp=sharing) are stored in `checkpoints/chexzero_weights/`, or the `model_dir` that is specified in the notebook.

## Evaluation

Given a numpy array of predictions (obtained from zero-shot inference), and a numpy array of ground truth labels, one can evaluate the performance of the model using the following code:

```python
import zero_shot
import eval

# loads in ground truth labels into memory
test_pred = y_pred_avg
test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

# evaluate model, no bootstrap
cxr_results: pd.DataFrame = eval.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset

# boostrap evaluations for 95% confidence intervals
bootstrap_results: Tuple[pd.DataFrame, pd.DataFrame] = eval.bootstrap(test_pred, test_true, cxr_labels) # (df of results for each bootstrap, df of CI)

# print results with confidence intervals
print(bootstrap_results[1])
```

The results are represented as a `pd.DataFrame` which can be saved as a `.csv`.

### CheXpert Test Dataset

In order to replicate the results in the paper, zero-shot inference and evaluation can be performed on the now publicly available CheXpert test dataset.

1. Download labels at [cheXpert-test-set-labels](https://github.com/rajpurkarlab/cheXpert-test-set-labels/blob/main/groundtruth.csv) and image files from [Stanford AIMI](https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c) and save in the `./data` directory in `CheXzero/`. The test dataset images should have the following directory structure:

```
data/
├─ CheXpert/
│  ├─ test/
│  │  ├─ patient64741/
│  │  │  ├─ study1/
│  │  │  │  ├─ view1_frontal.jpg
│  │  ├─ .../
```

2. Run `run_preprocess.py` script with the following arguments:

```bash
python run_preprocess.py --dataset_type "chexpert-test" --cxr_out_path "./data/chexpert_test.h5" --chest_x_ray_path "./data/CheXpert/test/"
```

This should save a `.h5` version of the test dataset images which can be used for evaluation.

3. Open sample zero-shot [notebook](https://github.com/rajpurkarlab/CheXzero/blob/main/notebooks/zero_shot.ipynb) and run all cells. If the directory structure is set up correctly, then all cells should run without errors.

## Issues

Please open new issue threads specifying the issue with the codebase or report issues directly to ekintiu@stanford.edu.

## Citation

```bash
Tiu, E., Talius, E., Patel, P. et al. Expert-level detection of pathologies from unannotated chest X-ray images via self-supervised learning. Nat. Biomed. Eng (2022). https://doi.org/10.1038/s41551-022-00936-9
```

## License

The source code for the site is licensed under the MIT license, which you can find in the `LICENSE` file. Also see `NOTICE.md` for attributions to third-party sources.
