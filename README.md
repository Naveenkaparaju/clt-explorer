# CLT Explorer

Interactive Streamlit app demonstrating the Central Limit Theorem.

Pick any distribution (exponential, uniform, chi-squared, beta, poisson, bimodal), adjust sample size and number of samples, and watch the sample means converge to a normal distribution in real time.

## Features

- 6 distributions to choose from
- Adjustable sample size and number of samples
- Side-by-side comparison: original distribution vs sample means
- Shapiro-Wilk and Kolmogorov-Smirnov normality tests
- Convergence panel showing n = 2, 5, 30, 100

## Live App

[https://clt-explorer.streamlit.app]([https://clt-explorer.streamlit.app](https://clt-explorer-ay7sruplxgzdd9c4yq4je5.streamlit.app/))

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Built by Naveen Kaparaju | Feb 2025
