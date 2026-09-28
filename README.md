# SCPaT: Semantic‑Aware Graph Routing with Transfer Entropy for multivariate time series forecasting


## 📌 Overview
SCPaT is a semantic structured framework designed for multivariate time series forecasting. By decomposing time series into semantic units and organizing them through a transfer entropy based graph structure, SCPaT captures heterogeneous temporal patterns and higher order dependencies among local dynamics. With an importance aware routing mechanism, SCPaT adaptively assigns different semantic blocks to specialized experts, enabling accurate and robust forecasting across various domains forecasting.
- Semantic Vector Encoder: Transforms raw multivariate time series into semantic units by extracting multi scale temporal representations and preserving local pattern consistency.
- Transfer Entropy Graph Constructor: Quantifies directed dependency relationships among semantic units and constructs a dynamic semantic graph to capture higher order interactions.
- Importance Aware Routing Module: Dynamically assigns semantic blocks to specialized experts according to their semantic characteristics, enabling differentiated modeling of trends, fluctuations, and periodic patterns.


![](./assets/model.png)
## ⚙️ Prerequisites

Make sure your Python version is matched with the required version, then install the required packages with this command:

```
pip install -r requirements.txt
```

## 📁 Prepare Datastes
Start by fetching the required datasets.All the data can be easily obtained from [iTransformer](https://drive.google.com/file/d/1l51QsKvQPcqILT3DwfjCgx8Dsg2rpjot/view?usp=drive_link).Then, set up a separate folder called `./data`. 

## 🚀 Training 

All scripts can be found in the `./scripts` directory. If you want to train model with an input length of 192 on the ETTm1 dataset, you can run the following script:

```shell
sh ./scripts/Long_term_forecasting/ETTm1.sh
```
## 📈 Results
Multivariate long-term forecasting results over four prediction horizons, $H \in \{96, 192, 336, 720\}$, with the input length fixed at $L=96$. The best and second-best results are marked in **bold red** and <u>blue underline</u>.
![](./assets/long.png)
Multivariate short-term forecasting results over three prediction horizons, $H \in \{12, 24, 48\}$, with the input length fixed at $L=96$. The best and second-best results are marked in **bold red** and <u>blue underline</u>.
![](./assets/short.png)

## 🙏 Acknowledgement
Our sincere appreciation goes to the following repositories for sharing invaluable code and datasets:

- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [iTransformer](https://github.com/thuml/iTransformer)
