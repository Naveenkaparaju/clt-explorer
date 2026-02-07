import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Central Limit Theorem Explorer", layout="wide")

st.title("Central Limit Theorem — Interactive Demo")
st.markdown("""
The Central Limit Theorem (CLT) says that if you take enough random samples from *any* distribution 
and compute the mean of each sample, those means will follow a normal (bell-shaped) distribution — 
even if the original data isn't normal at all. This app lets you see that happen in real time.
""")

st.markdown("---")

# sidebar controls
st.sidebar.header("Settings")

dist_choice = st.sidebar.selectbox(
    "Pick a distribution",
    ["Exponential", "Uniform", "Chi-Squared", "Beta (skewed)", "Poisson", "Bimodal"]
)

sample_size = st.sidebar.slider("Sample size (n)", min_value=2, max_value=200, value=30, step=1)
num_samples = st.sidebar.slider("Number of samples", min_value=100, max_value=10000, value=1000, step=100)
seed = st.sidebar.number_input("Random seed (for reproducibility)", value=42, step=1)

np.random.seed(int(seed))

# generate data from chosen distribution
def draw_population(name, size=100000):
    if name == "Exponential":
        return np.random.exponential(scale=2.0, size=size)
    elif name == "Uniform":
        return np.random.uniform(low=0, high=10, size=size)
    elif name == "Chi-Squared":
        return np.random.chisquare(df=3, size=size)
    elif name == "Beta (skewed)":
        return np.random.beta(a=2, b=8, size=size)
    elif name == "Poisson":
        return np.random.poisson(lam=3, size=size)
    elif name == "Bimodal":
        left = np.random.normal(loc=2, scale=0.8, size=size // 2)
        right = np.random.normal(loc=8, scale=0.8, size=size // 2)
        return np.concatenate([left, right])

population = draw_population(dist_choice)

# take samples and compute means
sample_means = []
for _ in range(num_samples):
    samp = np.random.choice(population, size=sample_size, replace=True)
    sample_means.append(np.mean(samp))
sample_means = np.array(sample_means)

# layout: two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Original Distribution: {dist_choice}")
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.hist(population, bins=60, color="#FF7043", edgecolor="white", alpha=0.85, density=True)
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.set_title(f"{dist_choice} Distribution (population)")
    
    pop_skew = stats.skew(population)
    pop_kurt = stats.kurtosis(population)
    ax1.text(0.97, 0.95, f"skew = {pop_skew:.2f}\nkurtosis = {pop_kurt:.2f}",
             transform=ax1.transAxes, ha="right", va="top",
             fontsize=9, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.tight_layout()
    st.pyplot(fig1)
    
    st.markdown(f"""
    **Population stats:** mean = {np.mean(population):.3f}, std = {np.std(population):.3f}, 
    skewness = {pop_skew:.2f}
    
    {"This is clearly not a normal distribution." if abs(pop_skew) > 0.3 else "This is already fairly symmetric."}
    """)

with col2:
    st.subheader(f"Distribution of Sample Means (n={sample_size})")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(sample_means, bins=50, color="#42A5F5", edgecolor="white", alpha=0.85, density=True)
    
    # overlay a normal curve to show how close it is
    mu = np.mean(sample_means)
    sigma = np.std(sample_means)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), color="#E53935", linewidth=2, label="Normal fit")
    ax2.legend()
    
    ax2.set_xlabel("Sample Mean")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Sample Means ({num_samples} samples, n={sample_size} each)")
    
    means_skew = stats.skew(sample_means)
    means_kurt = stats.kurtosis(sample_means)
    ax2.text(0.97, 0.95, f"skew = {means_skew:.2f}\nkurtosis = {means_kurt:.2f}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=9, bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
    plt.tight_layout()
    st.pyplot(fig2)
    
    st.markdown(f"""
    **Sample means stats:** mean = {mu:.3f}, std = {sigma:.3f}, 
    skewness = {means_skew:.2f}
    
    {"The sample means are approximately normal — CLT in action." if abs(means_skew) < 0.3 else "Try increasing the sample size to see the means become more normal."}
    """)

# Q-Q plots (added via Codex collaboration)
st.markdown("---")
st.subheader("Q-Q Plots")
st.markdown("A Q-Q plot compares data against a theoretical normal distribution. Points on the diagonal = normal.")

col_qq1, col_qq2 = st.columns(2)

with col_qq1:
    fig_qq1, ax_qq1 = plt.subplots(figsize=(5, 4))
    stats.probplot(population[:5000], dist="norm", plot=ax_qq1)
    ax_qq1.set_title("Q-Q Plot: Original Population")
    ax_qq1.get_lines()[0].set(color="#FF7043", markersize=3, alpha=0.5)
    ax_qq1.get_lines()[1].set(color="#E53935", linewidth=2)
    plt.tight_layout()
    st.pyplot(fig_qq1)
    st.markdown("The original data deviates from the diagonal — it's not normal.")

with col_qq2:
    fig_qq2, ax_qq2 = plt.subplots(figsize=(5, 4))
    stats.probplot(sample_means, dist="norm", plot=ax_qq2)
    ax_qq2.set_title("Q-Q Plot: Sample Means")
    ax_qq2.get_lines()[0].set(color="#42A5F5", markersize=3, alpha=0.5)
    ax_qq2.get_lines()[1].set(color="#E53935", linewidth=2)
    plt.tight_layout()
    st.pyplot(fig_qq2)
    st.markdown("The sample means fall much closer to the diagonal — CLT at work.")

# normality test
st.markdown("---")
st.subheader("Normality Check")

shapiro_stat, shapiro_p = stats.shapiro(np.random.choice(sample_means, size=min(5000, len(sample_means)), replace=False))
ks_stat, ks_p = stats.kstest(sample_means, 'norm', args=(mu, sigma))

col3, col4 = st.columns(2)
with col3:
    st.metric("Shapiro-Wilk p-value", f"{shapiro_p:.4f}")
    if shapiro_p > 0.05:
        st.success("Can't reject normality (p > 0.05)")
    else:
        st.warning("Rejects normality — try a larger sample size")

with col4:
    st.metric("Kolmogorov-Smirnov p-value", f"{ks_p:.4f}")
    if ks_p > 0.05:
        st.success("Can't reject normality (p > 0.05)")
    else:
        st.warning("Rejects normality — try a larger sample size")

# how sample size affects convergence
st.markdown("---")
st.subheader("How Sample Size Affects Convergence")
st.markdown("Watch how the distribution of sample means changes as you increase n:")

fig3, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for idx, n in enumerate([2, 5, 30, 100]):
    means = [np.mean(np.random.choice(population, size=n)) for _ in range(2000)]
    axes[idx].hist(means, bins=40, color="#7E57C2", edgecolor="white", alpha=0.85, density=True)
    
    m, s = np.mean(means), np.std(means)
    x = np.linspace(m - 4*s, m + 4*s, 200)
    axes[idx].plot(x, stats.norm.pdf(x, m, s), color="#E53935", linewidth=1.5)
    axes[idx].set_title(f"n = {n}\nskew = {stats.skew(means):.2f}")
    axes[idx].set_xlabel("Sample Mean")
    if idx == 0:
        axes[idx].set_ylabel("Density")

plt.tight_layout()
st.pyplot(fig3)

st.markdown("""
**What's happening:** As sample size (n) grows, the distribution of sample means gets 
tighter and more bell-shaped — regardless of what the original distribution looks like. 
That's the Central Limit Theorem. By n=30 or so, you're usually pretty close to normal.
""")

st.markdown("---")
st.markdown("""
*Built by Naveen Kaparaju | Feb 2025 | 
[GitHub Repo](https://github.com/naveenkaparaju/clt-explorer)*
""")
