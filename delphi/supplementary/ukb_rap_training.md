# Reproducing Delphi on UKB RAP

To obtain your own Delphi checkpoint, you can train it in the UK Biobank Research Analysis Platform (RAP), with the total cost under £10.

The results will be slightly different from what is reported in the paper, both due to stochasticity in training and the fact that the underlying UKB data has been updated since the paper was published.

The training hyperparameters are also marginally different from the paper: we use fewer training iterations (150k vs 200k) as this allows faster training without sacrificing quality, and a larger `block_size` (96 vs 48) to fit even the longest UKB sequences, which would otherwise be truncated.

**Prerequisites:** a functioning UKB RAP account with phenotypic data dispensed. Log in at https://ukbiobank.dnanexus.com.


## 1. Data preprocessing

Delphi uses three data sources from UKB:
- **ICD-10 first-occurrence fields** (`f.130000`–`f.132262`) — primary diagnoses with dates
- **Cancer registry** (`f.40006` / `f.40005`) — cancer type and date as parallel instance arrays (up to 17 cancers per participant)
- **Demographics and lifestyle** — sex (`f.31`), BMI (`f.21001`), smoking (`f.1239`), alcohol (`f.1558`), assessed at recruitment

These are converted into a flat binary format: three `uint32` columns per row — `[participant_id, age_in_days, token_id]` — deduplicated per participant-token pair and sorted by participant, then age.

To obtain the `.bin` files:

1. Go to **Tools > JupyterLab > New JupyterLab**
2. Create an instance with the following parameters:
   - Priority: **High**
   - Cluster configuration: **Spark cluster**
   - Instance type: **mem1_hdd1_v2_x16**
   - Number of nodes: **2**
   - Duration: **2 hours**
   - Feature: **HAIL**
3. Start the environment and wait for it to initialise (can take up to 30 minutes)
4. Open a terminal in JupyterLab and clone the repo:
   ```bash
   git clone --depth=1 -b ukb-rap https://github.com/gerstung-lab/Delphi.git && rm -rf Delphi/.git
   ```
5. Open `Delphi/data/ukb_real_data/example_ukb_rap_convert_new.ipynb`
6. In the first cell, set `output_dir` to the path where data will be saved — this should be a path in your **RAP project filesystem**, not the local Jupyter path
7. Execute all cells, including the last one that exports the data to your project using dx upload (~10–15 minutes)
8. Verify that `train.bin` and `val.bin` appear in the expected location in the RAP UI
9. Terminate the JupyterLab instance (click **Terminate** in the RAP UI, not just close the tab)


## 2. Training

Training requires a GPU and takes ~3 hours.

1. Go to **Tools > JupyterLab > New JupyterLab**
2. Create an instance:
   - Priority: **High**
   - Cluster configuration: **Single node**
   - Instance type: **mem2_ssd2_gpu1_v2_x16**
   - Duration: **6 hours**
   - Feature: **ML**
3. Start the environment, wait for it to initialise and open a terminal
4. RAP project data is mounted read-only at `/mnt/project/`; copy the repo to a writable location:
   ```bash
   cp -R /mnt/project/<my_delphi_location>/Delphi . && cd Delphi
   pip install -r requirements.txt
   ```
5. Start training:
   ```bash
   python train.py config/train_delphi.py --device=cuda --out_dir=Delphi-2M
   ```
   Optionally enable Weights & Biases logging with `--wandb_log=True`.
6. The best checkpoint is saved to `Delphi-2M/ckpt.pt` based on validation loss
7. Upload results back to RAP:
   ```bash
   dx upload -r /opt/notebooks/Delphi --destination <my_delphi_location>/
   ```
  The trailing `/` is essential!


## 3. Analyses

All notebooks can be run on the same GPU instance after training. Make sure to set the following parameters at the top of each notebook:
- Point to the real data (`dataset = 'ukb_real_data'`)
- Use full validation set (`dataset_subset_size = len(val_p2i)`)

### General evaluation (`evaluate_delphi.ipynb`)

Prediction accuracy (AUC vs age-sex baseline), calibration curves, attention patterns, and UMAP of learned disease embeddings. Optionally evaluate AUCs for all diseases.

### SHAP analysis (`shap_analysis.ipynb` and `shap-agg-eval.py`)

`shap_analysis.ipynb` computes per-patient SHAP values showing which input tokens drive predictions for specific diseases.

For dataset-wide aggregation, use the CLI script:

```bash
python shap-agg-eval.py \
  --checkpoint_dir Delphi-2M \
  --data_dir data/ukb_real_data \
  --device cuda \
  --output shap_agg.pickle
```

Running on the full validation set takes up to 10 hours. To get results faster, start with a subset:

```bash
python shap-agg-eval.py --checkpoint_dir Delphi-2M --data_dir data/ukb_real_data --device cuda --n 30000
```

Output: `shap_agg.pickle` — aggregated disease-disease interaction strengths.

### Trajectory sampling (`sampling_trajectories.ipynb`)

Generates synthetic patient trajectories using competing exponentials and compares them to real data (disease distributions, age distributions, incidence rates). Requires GPU, takes ~30 minutes.

### Uploading results

After all analyses are complete, upload everything back to RAP:

```bash
dx upload -r /opt/notebooks/Delphi --destination <my_delphi_location>/
```

Then terminate the JupyterLab instance.
