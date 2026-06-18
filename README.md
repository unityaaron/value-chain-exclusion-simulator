# 🌾 Smallholder Agricultural Yield Prediction & Causal Inference Pipeline
### Developed by Unity Aaron  
🔗 **Live Interactive Web Application:** [Click Here to Launch the App](https://unityaaron-machine-learning-research-live-app.streamlit.app/)

An end-to-end data science and machine learning system engineered to analyze, predict, and optimize smallholder farmer crop yields in Edo State. This repository transitions an initial exploratory analysis into a production-ready, automated predictive pipeline featuring advanced feature engineering and an interactive web deployment interface.

---

## 📊 Project Architecture & System Overview
The project is built using a modular framework designed for scalability and professional code separation:

* **`exploratory_analysis/`**: Contains the statistical identification of data structures, distribution shifts, and bimodal yield characteristics.
* **`statistical_inference/`**: Houses multiple OLS regression modeling used to isolate the true treatment effects of agricultural extension training while strictly controlling for confounding physical resource factors (Farm Size, Input Volumes).
* **`machine_learning_model/`**: Houses the predictive engines, including the optimized Random Forest Regressor champion model, cross-validation scripts, and frozen deployment assets (`.pkl`).
* **`run_pipeline.py`**: An automated, single-click ETL pipeline that ingests raw operational spreadsheets, runs category mapping, executes feature calculations, and exports live predictions.
* **`app.py`**: A public-facing interactive web application allowing non-technical stakeholders to interface with the predictive model in real time.

---

## 🧠 Key Analytical Discoveries & Performance Metrics

### 1. Causal Inference Check (Combating the Umbrella Illusion)
Through multiple linear regression, the intervention isolated the standalone impact of extension training. The training treatment effect remained statistically robust ($p < 0.001$) even when physical factor endowments were mathematically controlled, proving that yield gains are actively driven by capacity building rather than baseline farmer resource wealth.

### 2. Feature Engineering Multiplier Effect
By introducing an engineered interaction variable ($\text{Fertilizer Used} \times \text{Training Attended}$), the Random Forest model isolated the combined impact of physical inputs and technical knowledge. 
* **Primary Predictive Driver**: The interaction feature achieved the single highest predictive weight (**40.5% Feature Importance**), proving that fertilizer efficiency is structurally unlocked when combined with extension education.

### 3. Model Validation Stability
To guard against lucky data splits, the system was validated using a strict **5-Fold Cross-Validation** protocol:
* **Out-of-Sample Performance**: Achieved a cross-validated **Average Mean Absolute Error (MAE) of 2.41%**.
* **System Consistency**: Recorded an exceptionally low **Standard Deviation of 0.37%**, demonstrating immense model stability across varying farmer sub-populations.

---

## 🛠️ Installation & Operational Deployment

Follow these terminal instructions on your local machine to deploy the automated pipeline and interface dashboard:

### 1. Clone the Architecture
```bash
git clone [https://github.com/unityaaron/DATA_SCIENCE_MACHINE_LEARNING.git](https://github.com/unityaaron/DATA_SCIENCE_MACHINE_LEARNING.git)
cd DATA_SCIENCE_MACHINE_LEARNING
```  
### 2. Run the Automated Production Pipeline ###  
To process raw farmer documents and export an automated predictions spreadsheet automatically, execute:  
```bash
python run_pipeline.py  
```
### 3. Launch the Interactive Web Dashboard
To deploy the live local user interface on your browser, execute:
```bash  
streamlit run app.py
```