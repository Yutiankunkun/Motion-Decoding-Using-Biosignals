# Motion Decoding Using Biosignals with Reservoir Computing for Skateboard Trick Classification

---

## Introduction

In the field of brain computer interfaces (BCIs), the decoding of motion-related brain signals has demonstrated significant potential for various applications including sports training and rehabilitation. 

This work focuses on **skateboard trick classification** using **EEG-based biosignals** and **reservoir computing** techniques. By leveraging data collected during different skateboard maneuvers, we aim to investigate how well machine learning architectures, particularly those that incorporate reservoir computing, can distinguish between frontside/backside kickturns and pumping movements under both **within-subject** and **cross-subject** conditions.

---

## Results

We present our **within-subject** and **cross-subject** classification accuracies below. For additional metrics (precision, recall, and F1-score), refer to the Jupyter notebooks in the provided directories.

### Within-subject Results

| Subject   | EEGNet            | DeepCNN          | ShallowCNN       | FBCNet            | EEGConformer      | ESNNet            |
|-----------|-------------------|------------------|------------------|-------------------|-------------------|-------------------|
| subject0  | 79.25% ± 4.40%    | 79.25% ± 1.60%   | 68.81% ± 1.58%   | 71.32% ± 1.81%    | 80.63% ± 3.06%    | 83.02% ± 2.09%    |
| subject1  | 80.25% ± 1.97%    | 82.64% ± 0.72%   | 66.29% ± 0.72%   | 79.87% ± 0.77%    | 81.38% ± 2.46%    | 83.14% ± 1.21%    |
| subject2  | 74.81% ± 4.74%    | 75.32% ± 2.69%   | 69.62% ± 1.18%   | 78.61% ± 1.37%    | 75.06% ± 3.46%    | 78.61% ± 1.13%    |
| subject3  | 87.62% ± 1.56%    | 89.25% ± 0.52%   | 79.25% ± 1.43%   | 88.62% ± 1.35%    | 89.38% ± 2.03%    | 89.50% ± 1.03%    |
| subject4  | 77.62% ± 3.43%    | 81.50% ± 1.85%   | 69.88% ± 1.12%   | 75.50% ± 1.56%    | 78.88% ± 2.48%    | 82.00% ± 2.09%    |

### Cross-subject Results

| Model        | Accuracy           |
|--------------|--------------------|
| EEGNet       | 50.43% ± 2.18%     |
| DeepCNN      | 49.34% ± 6.09%     |
| ShallowCNN   | 51.06% ± 2.15%     |
| FBCNet       | 50.73% ± 1.18%     |
| EEGConformer | 51.06% ± 2.66%     |
| ESNNet       | 51.26% ± 2.32%     |

For more details about metrics, please check IPython notebooks:
- [Within-subject Results](./within-subject/results/all_results.csv)
- [Cross-subject Results](./cross-subject/results/all_results.csv)

---

## Training Data

Unzip `data.zip` to access training data for each participant (denoted as `subject0` through `subject4`). Each participant's folder contains three `.mat` files (`train1.mat`, `train2.mat`, `train3.mat`). The directory structure is as follows:

```bash
train
├── subject0
│   ├── train1.mat
│   ├── train2.mat
│   └── train3.mat
└── ...
```

---

## Training Data Details

Each `.mat` file contains the following keys:

- **`times`** (in milliseconds): Time points sampled at 2 ms intervals (500 Hz).
- **`data`**: A multi-dimensional array of EEG signals in microvolts, organized as (channels x samples).
- **`ch_labels`**: The names of all 72 channels based on the [International 10-10 system](https://commons.wikimedia.org/wiki/File:EEG_10-10_system_with_additional_information.svg).
- **`event`**: Contains the fields `init_index`, `type`, and `init_time`, which indicate the index of the event, the trick type, and the onset time of the trick in seconds, respectively.

The **type** field in `event` corresponds to one of the following skateboard maneuvers:

| Trick Type | Description                                           |
|------------|-------------------------------------------------------|
| 11         | Frontside kickturn toward the LED-side ramp           |
| 12         | Backside kickturn toward the LED-side ramp            |
| 13         | Pumping toward the LED-side ramp                      |
| 21         | Frontside kickturn toward the Laser-side ramp         |
| 22         | Backside kickturn toward the Laser-side ramp          |
| 23         | Pumping toward the Laser-side ramp                    |

For details on how to load this data, please see the `tutorial.ipynb` notebook in the repository.

---

## Stance File

The stance file (`stance.csv`) contains the **board stance** for each participant, specifying how they position themselves on the board (e.g., which foot is forward). 

By incorporating stance information into the classification model, researchers may observe improved discrimination between frontside and backside maneuvers.

---

## Conclusion and Future Work

This dataset and benchmark provide valuable insights into **EEG-based motion decoding** for skateboarding maneuvers. The comparison of different neural network architectures—ranging from simple convolutional models to advanced transformers and reservoir-based networks—demonstrates the challenges of **cross-subject** generalization.

### Future Work
Future work may involve:
- **Transfer learning**
- **Data augmentation**
- **Multi-modal integration** (e.g., kinematic or electromyographic data)

These approaches could improve classification performance and robustness in real-world conditions.