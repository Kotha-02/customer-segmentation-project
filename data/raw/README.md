# Raw data

The raw dataset (`online_retail.csv`, ~40MB, ~540K rows) is not committed to
keep the repo lightweight. The processed, ready-to-use outputs the app
actually reads are already included in `data/processed/`.

To regenerate everything from scratch:

1. Download the **Online Retail II** dataset from the UCI Machine Learning
   Repository: https://archive.ics.uci.edu/dataset/502/online+retail+ii
   (or the Kaggle mirror: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
2. Save it as `data/raw/online_retail.csv` with columns:
   `Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country`
3. Run:
   ```bash
   python src/pipeline.py
   ```
   This writes `data/processed/customer_segments.csv`,
   `data/processed/segment_summary.csv`, and `data/processed/k_selection.csv`.
