# Kidney Stone Detection with YOLOv8

This project trains a YOLOv8 object-detection model to locate kidney stones in medical images. The complete workflow—dataset download, exploratory visualisation, training, evaluation, and inference—is contained in the [Jupyter notebook](KIDNEY_STONE_DETECTION_USING_YOLO_V8.ipynb).

> **Important:** This is an educational machine-learning project, not a clinical diagnostic tool. Do not use its predictions to make medical decisions.

## What the notebook does

- Downloads the Kaggle kidney-stone image dataset.
- Displays annotated training samples.
- Fine-tunes a pretrained `yolov8x.pt` detector for 50 epochs.
- Plots training losses, precision, recall, and mAP metrics.
- Evaluates the best checkpoint on the test split.
- Runs predictions on randomly selected test images.

## Requirements

The notebook is written for **Google Colab** and Python 3. Install the required packages:

```bash
pip install ultralytics squarify matplotlib opencv-python pandas seaborn kaggle
```

You will also need a Kaggle account and API token (`kaggle.json`) to download the dataset.

## Dataset setup

The notebook downloads the [Kidney Stone Image Dataset](https://www.kaggle.com/datasets/safurahajiheidari/kidney-stone-images):

```bash
kaggle datasets download -d safurahajiheidari/kidney-stone-images
unzip kidney-stone-images.zip
```

For Colab, upload `kaggle.json` when prompted and place it in `~/.kaggle/` with restricted permissions:

```bash
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

After extraction, the notebook expects a YOLO-format dataset similar to:

```text
/content/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

If you run locally instead of in Colab, update the `/content/...` paths in the notebook to match your machine.

## Run the project

1. Open `KIDNEY_STONE_DETECTION_USING_YOLO_V8.ipynb` in Google Colab or Jupyter.
2. Install the dependencies and configure your Kaggle API token.
3. Run the cells in order to download and extract the dataset.
4. Confirm that `data.yaml` points to the train, validation, and test image folders.
5. Run the training cell:

   ```python
   from ultralytics import YOLO

   model = YOLO("yolov8x.pt")
   model.train(data="/content/data.yaml", seed=42, epochs=50, lr0=0.001)
   ```

6. Evaluate the best trained weights:

   ```python
   model = YOLO("/content/runs/detect/train/weights/best.pt")
   metrics = model.val(conf=0.25, split="test")
   ```

Training outputs, including `results.csv`, metric plots, the confusion matrix, and `best.pt`, are written under `/content/runs/detect/train/` by default.

## Repository contents

| File | Description |
| --- | --- |
| `KIDNEY_STONE_DETECTION_USING_YOLO_V8.ipynb` | End-to-end Colab/Jupyter workflow for training, evaluation, and inference. |
| `README.md` | Project setup and usage guide. |

## Notes

- `yolov8x.pt` is the largest standard YOLOv8 checkpoint and generally needs a GPU runtime. For faster, lower-resource experiments, replace it with `yolov8n.pt` or another YOLOv8 checkpoint.
- Results vary with the dataset version, hardware, package versions, and training configuration. The repository does not include pretrained weights or a published benchmark score.

## License

No license file is currently included in this repository. Contact the repository owner before reusing the code or trained outputs outside personal or educational use.
