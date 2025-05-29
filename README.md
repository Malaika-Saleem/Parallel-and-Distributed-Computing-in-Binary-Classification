# Parallel and Distributed Computing Project: ML Model Acceleration

## Project Overview

This project investigates and compares various techniques for accelerating the training of machine learning models, specifically focusing on a binary classification task. The primary goal is to analyze the performance (execution time and accuracy) improvements achieved through parallel computing, distributed systems, GPU acceleration, and hybrid approaches compared to serial execution. The project uses a synthetic dataset and evaluates a `RandomForestClassifier` as the primary model for benchmarking acceleration techniques, alongside other standard classifiers for baseline comparison.

## Dataset

The dataset used is `pdc_dataset_with_target.csv`, located in the `/content/sample_data/` directory within the notebook's environment. It's a synthetic dataset with:
*   7 features (a mix of numerical and categorical)
*   1 binary target variable
*   Initially 41,000 rows, reduced to 33,416 after dropping rows with NaN values.

**Preprocessing Steps:**
1.  **Handling Missing Values:** Rows with any NaN values are dropped.
2.  **Encoding:** Categorical features (`feature_3`, `feature_5`) are one-hot encoded using `pd.get_dummies(drop_first=True)`.
3.  **Normalization:** Numerical features are scaled using `StandardScaler`.
4.  **Balancing:** The dataset is balanced using `SMOTE` (Synthetic Minority Over-sampling Technique) to handle class imbalance in the target variable.
5.  **Data Splitting:** The data is split into an initial small set (`x_init`, `y_init` - 15% of data) for quick accelerator testing, and a larger set (`X_train`, `X_test`, `y_train`, `y_test`) for final model evaluation.

## Technologies & Libraries

*   **Core Python & Data Handling:**
    *   Python 3.x
    *   Pandas, NumPy
    *   Matplotlib, Seaborn (for visualization)
    *   OS, Time
*   **Machine Learning:**
    *   Scikit-learn (RandomForestClassifier, LogisticRegression, SVC, DecisionTreeClassifier, VotingClassifier, StandardScaler, MLPClassifier, metrics, model_selection)
    *   XGBoost (XGBClassifier)
    *   CatBoost (CatBoostClassifier)
    *   LightGBM (LGBMClassifier)
    *   Imbalanced-learn (SMOTE)
*   **Parallel Computing (CPU):**
    *   `multiprocessing.Pool`
    *   `concurrent.futures.ProcessPoolExecutor`
    *   `concurrent.futures.ThreadPoolExecutor`
    *   Joblib (used implicitly by scikit-learn with `n_jobs=-1`)
*   **Distributed Systems:**
    *   Dask (`dask.delayed`, `dask.distributed.Client`)
    *   Apache Spark (`pyspark.sql.SparkSession`, `pyspark.SparkContext`)
    *   *MPI (Message Passing Interface) - Mentioned as being implemented in a separate `.py` file, not directly in the notebook.*
*   **GPU Acceleration:**
    *   TensorFlow (Keras API for Dense layers, Adam optimizer; also `tf.convert_to_tensor`)
    *   PyTorch (`torch.nn`, `torch.optim`, `torch.tensor`, device management)
    *   CuPy (`cupy.asarray`) - primarily for GPU-accelerated NumPy-like operations.

## Setup and Installation

1.  **Clone the repository (if applicable) or download the notebook and dataset.**
2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv pdc_env
    source pdc_env/bin/activate  # On Windows: pdc_env\Scripts\activate
    ```
3.  **Install dependencies:**
    The notebook installs `catboost` directly. A `requirements.txt` file would be beneficial. Based on imports, a minimal `requirements.txt` would look like this:
    ```
    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    xgboost
    catboost
    lightgbm
    imbalanced-learn
    tensorflow
    torch
    # For Dask with distributed capabilities
    dask[distributed]
    # For PySpark
    pyspark
    # For CuPy (requires CUDA toolkit installed separately)
    cupy-cudaXX # Replace XX with your CUDA version, e.g., cupy-cuda118 or cupy-cuda12x
    joblib # Though often a scikit-learn dependency
    ```
    Install using:
    ```bash
    pip install -r requirements.txt
    ```
    *   **Note on GPU Libraries:** For TensorFlow (GPU), PyTorch (GPU), and CuPy, you need a compatible NVIDIA GPU, NVIDIA drivers, and the CUDA toolkit installed on your system. Installation instructions for these libraries often vary based on your CUDA version.
    *   **Note on PySpark:** Requires Java Development Kit (JDK) to be installed and `JAVA_HOME` environment variable set.
    *   **Note on MPI:** To run MPI-based experiments (mentioned as external), an MPI implementation like OpenMPI or MPICH needs to be installed.

4.  **Dataset:**
    Ensure `pdc_dataset_with_target.csv` is placed in a path accessible by the notebook. The notebook uses `/content/sample_data/pdc_dataset_with_target.csv`, which is typical for Google Colab. Adjust this path if running locally.

## Running the Code

1.  Activate your Python environment.
2.  Launch Jupyter Notebook or JupyterLab:
    ```bash
    jupyter notebook PDC_project_22i0509_22i0612.ipynb
    # or
    jupyter lab PDC_project_22i0509_22i0612.ipynb
    ```
3.  Open the `PDC_project_22i0509_22i0612.ipynb` notebook.
4.  Run the cells sequentially.
    *   Cells involving PySpark, Dask distributed clients, or GPU libraries might have specific setup or hardware requirements.
    *   The notebook uses a smaller subset of data (`x_init_train`, `y_init_train`) for benchmarking different accelerator techniques to ensure quicker iterations.
    *   The final model evaluation and comparison (Serial vs. Parallel CPU/GPU) uses a larger training set (`X_train`, `y_train`).

## Project Structure (Implicit)

*   `PDC_project_22i0509_22i0612.ipynb`: The main Jupyter notebook containing all the code and analysis.
*   `/content/sample_data/pdc_dataset_with_target.csv`: The input dataset.
*   *(mpi_script.py)*: (Mentioned but not provided) A separate Python script for MPI-based experiments.

## Key Findings and Results

The project systematically compares the execution time of training a `RandomForestClassifier` (and other models for baseline) using various acceleration techniques.

1.  **Baseline Model Performance (on `x_init` data):**
    *   XGBoost: ~0.5356 accuracy
    *   Voting Ensemble (RF + CatBoost): ~0.5215 accuracy
    *   Other models (LR, SVM, DT, RF, CatBoost, LightGBM, MLP) showed accuracies mostly in the range of 0.47 to 0.52.

2.  **Accelerator Performance (Training `RandomForestClassifier` 4 times on `x_init_train`):**
    *   **Parallel Computing (CPU):**
        *   Multiprocessing (`ProcessPoolExecutor`, `n_jobs=-1` in RF): ~7.49 seconds
        *   Multithreading (`ThreadPoolExecutor`, RF `n_jobs=1`): ~3.61 seconds. *Multithreading was notably faster here, possibly due to GIL release by scikit-learn for this task or overhead in process creation/communication for multiprocessing with this specific workload.*
    *   **Distributed Systems:**
        *   Dask (`dask.delayed`, scheduler="threads"): ~4.64 seconds
        *   PySpark: ~9.41 seconds. *Dask performed better than PySpark in this setup.*
        *   MPI: Not timed in the notebook, external.
    *   **GPU Acceleration (Data prep and model training orchestration, RF itself is CPU-bound in scikit-learn):**
        *   TensorFlow (tensor conversion, RF on numpy arrays): ~0.95 seconds
        *   PyTorch (tensor conversion, RF on numpy arrays): ~0.92 seconds. *Slightly faster.*
        *   CuPy (data prep, RF on numpy arrays): ~2.21 seconds (includes CuPy array conversions).
    *   **Hybrid Models (Training RF once with different configurations on `x_init_train`):**
        *   Dask + Multithreading (joblib backend for RF `n_jobs=-1`): ~1.45 seconds
        *   Multithreading + PyTorch-style data flow: ~4.25 seconds (for 4 "epochs" or RF trainings)
        *   Dask + PyTorch-style data flow (simulating epochs with Dask tasks): ~12.54 seconds (for 4 RF models with varying estimators)
        *   All 3 (Torch-style data + Multithreaded chunk processing + Dask for overall parallelism): ~2.64 seconds

3.  **Final Conclusion on Best Accelerator (from notebook):**
    *   The notebook concludes: *"SO we can conclude that the best is TORCH"* and *"Multithreading + Dask + Torch performed the best"* (referring to the "All 3" hybrid approach with 2.64s). This likely refers to the efficiency of PyTorch for data handling (especially with potential GPU interplay for data movement) and the combination of Dask for high-level task parallelism and multithreading for finer-grained parallelism within tasks.

4.  **Serial vs. Parallel Execution (Full `X_train` dataset, `RandomForestClassifier`):**
    *   **CPU Serial (`n_jobs=1`):** ~17.45 seconds (Accuracy: ~0.63)
    *   **GPU Serial (Implicitly still CPU for RF, but on a GPU-enabled machine, `n_jobs=1`):** ~6.16 seconds (Accuracy: ~0.63). *Improvement: ~64.70% over CPU serial.* This speedup on a "GPU machine" even for serial CPU code can sometimes be due to faster I/O or a more powerful CPU often paired with GPUs.
    *   **Parallel (PyTorch-style data flow, RF `n_jobs=-1`, effectively multithreaded on CPU):** ~4.67 seconds (Accuracy: ~0.63). *Improvement: ~73.24% over CPU serial.*

**Overall Performance Highlights:**
*   Leveraging PyTorch for data handling (even if the core ML model is scikit-learn's CPU-bound RandomForest) combined with multithreading (`n_jobs=-1`) and potentially Dask for higher-level orchestration, provides significant speedups.
*   GPU environments (even if not directly executing all model parts on GPU) can offer speed benefits due to better I/O or associated CPU capabilities.
*   Multithreading was more effective than multiprocessing for the RandomForest training task in this specific context.

## Future Work

*   Implement and benchmark MPI-based parallel training for `RandomForestClassifier`.
*   Explore GPU-native implementations of Random Forests (e.g., cuML's RandomForest from RAPIDS.ai) for direct GPU execution.
*   Conduct more rigorous hyperparameter tuning for all models.
*   Test on larger datasets to see how different parallelization strategies scale.
*   Profile memory usage in addition to execution time.
*   Extend the comparison to other types of models (e.g., Neural Networks where GPU benefits are more direct and pronounced).
