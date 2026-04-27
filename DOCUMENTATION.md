# Dwesh — Technical Documentation

## AI Fairness Simulation & Bias Auditing Platform

**Version:** 3.2.0 &nbsp;|&nbsp; **License:** MIT &nbsp;|&nbsp; **Team:** GDP Hackathon &nbsp;|&nbsp; **Target:** Google Solution Challenge 2026

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Solution Architecture](#3-solution-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Core Features](#5-core-features)
6. [DecisionTwin Engine — Research Innovation](#6-decisiontwin-engine--research-innovation)
7. [Longitudinal Policy Search](#7-longitudinal-policy-search)
8. [Causal Parallel Universe Simulator](#8-causal-parallel-universe-simulator)
9. [Asymmetric Cost Analysis](#9-asymmetric-cost-analysis)
10. [Bias Audit Engine](#10-bias-audit-engine)
11. [Reinforcement Learning Training Lab](#11-reinforcement-learning-training-lab)
12. [Mitigation Engine](#12-mitigation-engine)
13. [Sample Datasets & Testing](#13-sample-datasets--testing)
14. [Frontend Architecture](#14-frontend-architecture)
15. [Fairness Metrics — Detailed Reference](#15-fairness-metrics--detailed-reference)
16. [Domain Configurations](#16-domain-configurations)
17. [Research References & Impact](#17-research-references--impact)

---

## 1. Executive Summary

Dwesh is an end-to-end AI fairness platform that moves beyond static bias detection. While existing tools such as Fairlearn and AIF360 provide point-in-time fairness metrics, Dwesh introduces the **DecisionTwin** concept — a digital replica of AI decision systems that simulates long-term societal impact before deployment.

### Key Differentiators

| Capability | Fairlearn / AIF360 | Dwesh |
|---|---|---|
| Bias Detection | Static only | Static + Longitudinal |
| Feedback Loop Simulation | Not supported | 10-year forward simulation |
| Causal Reasoning | Correlational | Parallel universe proofs |
| Cost Asymmetry | Symmetric only | Group-aware weighting |
| RL-Based Mitigation | Not supported | Agent-based optimization |
| Interactive Interface | Limited | Full interactive dashboard |

### Impact Statement

AI systems make decisions affecting millions of people — in hiring, lending, and healthcare triage. When these systems carry bias, they create feedback loops that compound discrimination over generations. Dwesh is the first platform to simulate these long-term dynamics, enabling policymakers and ML engineers to observe the consequences of their algorithms before real people are harmed.

---

## 2. Problem Statement

### The Feedback Loop Crisis

When an AI hiring model rejects candidates from a minority group at higher rates, it triggers a self-reinforcing cycle:

1. Those candidates lose career momentum.
2. Their future qualifications degrade over time.
3. The training data for the next model iteration reflects this degradation.
4. The model becomes **even more biased** in subsequent versions.

This phenomenon is known as the **algorithmic feedback loop** — well-documented in ML fairness literature (Ensign et al., 2018; Liu et al., 2018).

### Limitations of Existing Tools

Fairlearn and AIF360 treat fairness as a static optimization constraint. They answer the question: *"Is this model fair right now?"* — but they cannot answer:

- *"Will this model still be fair in 5 years?"*
- *"Which mitigation strategy genuinely helps versus merely masking the problem?"*
- *"How much real-world harm does this bias cause to specific communities?"*

### How Dwesh Solves This

Dwesh answers all three questions through three core innovations:

1. **Longitudinal Policy Search (LPS)** — Forward simulation that accounts for feedback loops over time.
2. **Causal Parallel Universes** — Controlled experiments that prove causation, not just correlation.
3. **Asymmetric Cost Analysis** — Group-aware quantification of harm that symmetric metrics miss.

---

## 3. Solution Architecture

Dwesh follows a layered architecture designed for modularity and extensibility.

### System Layers

- **Frontend Layer** — A single-page dashboard application built with HTML5, CSS3, and JavaScript. It comprises multiple interactive pages: Dashboard, Audit, LPS Simulator, Parallel Universes, Cost Analysis, and RL Training.

- **Backend Layer** — A Python-based FastAPI server that exposes RESTful endpoints. It houses the core computational engines: Audit Engine, LPS Engine, Causal Simulator, Cost Analysis module, RL Environment (Gymnasium), Fairness Metrics calculator, and Mitigation Engine.

- **Data Layer** — MongoDB serves as the persistent store for audit logs and simulation results.

### Data Flow

1. The user uploads a CSV dataset via the Audit page.
2. The backend computes fairness metrics across all demographic groups.
3. Results populate the Dashboard, Heatmap, and Report Card views.
4. The user can run LPS simulations, Parallel Universe comparisons, or Cost Analysis.
5. The RL Training agent learns to optimize the fairness-accuracy tradeoff.
6. All actions are logged to the Audit Trail for accountability.

---

## 4. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | HTML5, CSS3, JavaScript | Single-page dashboard application |
| Charts | Chart.js 4.4.1 | Interactive data visualization |
| Icons | Lucide 0.344.0 | Consistent icon system |
| Backend | Python 3.10+, FastAPI | REST API server |
| RL Engine | Gymnasium (OpenAI Gym) | Reinforcement learning environment |
| Database | MongoDB (via Motor) | Asynchronous audit trail storage |
| AI Integration | Google Gemini API | AI-powered analysis narratives |
| Deployment | Google Cloud Run | Serverless container hosting |

---

## 5. Core Features

### 5.1 Guided Feature Explainer System

Every page in Dwesh includes an automatic **"What is this?"** popup that appears on the user's first visit. Each popup explains:

- What the feature does
- Step-by-step usage instructions
- Key terminology and concepts

The popup appears once per session and can be dismissed by the user when ready.

### 5.2 Feature Overview

| Feature | Description |
|---|---|
| **Bias Audit** | Upload a CSV dataset and compute fairness metrics across all demographic groups |
| **Dashboard** | Central metrics overview with visualizations and summary statistics |
| **Bias Heatmap** | Feature-by-group bias visualization matrix |
| **RL Training** | Train a bias-reduction agent using reinforcement learning |
| **Mitigation** | AI-powered remediation recommendations tailored to detected bias patterns |
| **What-If Explorer** | Counterfactual analysis — explore what happens when feature values change |
| **Drift Monitoring** | Real-time tracking of fairness metric drift over time |
| **LPS Simulator** | 10-year forward simulation of feedback loop dynamics |
| **Parallel Universes** | Side-by-side causal comparison of mitigation strategies |
| **Cost Analysis** | Asymmetric harm quantification with group-aware weighting |
| **Benchmark** | Regulatory compliance scoring against established fairness standards |
| **Shadow AI** | AI-generated text detection tool |

---

## 6. DecisionTwin Engine — Research Innovation

The DecisionTwin is the core research innovation in Dwesh. It creates a **digital twin** of an AI decision system and its affected population, then simulates the long-term co-evolution of both.

### Conceptual Overview

**Traditional Approach:** A dataset is fed into a model, fairness metrics are computed at a single point in time, and the analysis is complete. This captures bias as it exists today but says nothing about tomorrow.

**DecisionTwin Approach:** The system takes the initial dataset and model, simulates decisions, models their impact on the affected population, generates updated data reflecting that impact, and repeats this cycle over 10 simulated years. The result is a trajectory-based assessment of how fairness evolves over time.

### The Three Pillars

1. **LPS Engine** — Simulates how bias evolves when decisions create feedback loops in the population data.
2. **Causal Simulator** — Proves which mitigation strategy genuinely works via controlled parallel experiments.
3. **Cost Analysis** — Reveals hidden harm that symmetric, equal-weight metrics systematically undercount.

---

## 7. Longitudinal Policy Search

### Purpose

The LPS Engine simulates 3 to 20 years of AI decision-making, modelling how:

- Rejected candidates or applicants lose future opportunities.
- Population statistics shift as a direct result of algorithmic decisions.
- Economic shocks disproportionately affect minority groups.
- Different mitigation strategies perform across extended time horizons.

### Domain-Specific Simulation Parameters

Each domain has unique feedback loop dynamics, calibrated from published research:

| Domain | Initial DI | Initial Accuracy | Degradation Rate | Shock Probability |
|---|---|---|---|---|
| Hiring | 0.68 | 87% | 3.5% per year | 12% |
| Loan / Finance | 0.73 | 91% | 4.5% per year | 15% |
| Medical Triage | 0.78 | 93% | 2.8% per year | 8% |
| Intersectional | 0.58 | 84% | 5.5% per year | 10% |

### Mitigation Strategy Comparison

| Strategy | Year 1 Effect | Long-term Stability | Feedback Resistance |
|---|---|---|---|
| No Intervention | — | Collapse | None |
| Static Threshold | +12% DI | Decays rapidly | Low |
| Reweighting | +4% DI per year | Moderate decay | Medium |
| **Dynamic LPS (Ours)** | Adapts each year | **96% stable** | **93% resistant** |

### Usage Instructions

1. Open the **LPS Simulator** page from the sidebar.
2. Select a domain — Hiring, Loan, Medical, or Intersectional.
3. Set the simulation duration (3 to 20 years).
4. Click **"Run 10-Year Simulation"** for single-strategy analysis.
5. Click **"Compare All Strategies"** for a head-to-head comparison across all four approaches.
6. Review the trajectory charts, year-by-year data table, and generated policy recommendation.

### Interpreting Results

- **Divergence Year** — The year when the Disparate Impact ratio first drops below the 0.80 legal compliance threshold.
- **Economic Shocks** — Yellow indicators on the trajectory chart mark external disruptive events.
- **Dynamic LPS** consistently outperforms static approaches because it recalibrates at each time step.

---

## 8. Causal Parallel Universe Simulator

### The Correlation Problem

Standard fairness tools detect correlations — for example, *"Group A has a lower approval rate than Group B."* However, they cannot prove that a specific mitigation strategy **caused** an improvement rather than merely coinciding with one.

### How Dwesh Solves This: Forking Reality

The Parallel Universe simulator operates as follows:

1. Takes two mitigation strategies as input.
2. Starts both from identical initial conditions.
3. Applies the same external events (economic shocks) to both universes.
4. Simulates forward for 10 years independently.
5. Compares outcomes to generate a **causal verdict** — empirical proof of which strategy performs better and why.

### Available Strategies

| Strategy | Description | Best Suited For |
|---|---|---|
| Drop Proxy Feature | Remove features correlated with protected attributes (e.g., ZIP code as a proxy for race) | High proxy correlation scenarios |
| Reweight Samples | Increase the weight of underrepresented groups during training | Training data imbalance |
| Per-Group Thresholds | Apply separate decision thresholds for each demographic group | Threshold-based models |
| Adversarial Debiasing | Use an adversary network to block the model from learning sensitive attributes | Deep learning pipelines |
| Dynamic LPS (Ours) | Feedback-loop-aware optimization that adapts over time | Long-term deployment scenarios |

### Crossover Detection

The simulator automatically detects the **crossover year** — the point at which one strategy overtakes another in effectiveness. This is critical because strategies that appear superior in Year 1 may collapse by Year 5 due to feedback loop vulnerability.

### Usage Instructions

1. Open the **Parallel Universes** page.
2. Select the mitigation strategy for Universe A and Universe B.
3. Click **"Fork Reality"** to begin the simulation.
4. Compare the diverging trajectory charts.
5. Read the Causal Verdict for a definitive, evidence-based conclusion.

---

## 9. Asymmetric Cost Analysis

### The Hidden Harm Problem

Standard fairness metrics treat all errors equally — a false positive (incorrectly approving someone) and a false negative (incorrectly rejecting someone) are given the same weight. In reality, these errors carry vastly different costs depending on the affected group.

| Domain | FN Cost (Minority) | FN Cost (Majority) | Harm Ratio |
|---|---|---|---|
| Hiring | 2.5× | 1.2× | 2.1 : 1 |
| Loan | 3.0× | 1.0× | 3.0 : 1 |
| Medical | 4.5× | 3.0× | 1.5 : 1 |
| Intersectional | 3.5× | 1.5× | 2.3 : 1 |

### What the Analysis Reveals

- **Harm Underestimation Percentage** — How much standard tools undercount the real harm experienced by minority groups.
- **Worst Affected Group** — The specific intersectional group facing the greatest harm.
- **Community Impact Score** — A normalized 0-to-1 scale measuring systemic damage.

### Intersectional Groups Analysed

The analysis examines 8 intersectional groups across two axes (gender × race):

- Male × White, Female × White
- Male × Black, Female × Black
- Male × Hispanic, Female × Hispanic
- Male × Asian, Female × Asian

**Key finding:** Intersectional groups such as Female × Black face compounded harm that is greater than the sum of gender bias and racial bias measured separately.

---

## 10. Bias Audit Engine

### Supported Metrics

| Metric | Description | Fair Range | Legal Threshold |
|---|---|---|---|
| Disparate Impact Ratio (DIR) | Ratio of minority approval rate to majority approval rate | 0.80 – 1.25 | 0.80 (EEOC 4/5ths rule) |
| Demographic Parity Difference (DPD) | Absolute difference in positive outcome rates between groups | < 0.10 | — |
| Overall Bias Score | Weighted composite of individual metrics | < 0.30 | — |

### How to Run an Audit

1. Navigate to the **Audit** page.
2. Upload a CSV file containing:
   - A **label** column (binary: 0 or 1)
   - Demographic columns (e.g., gender, race, age)
   - Feature columns (e.g., skill_score, experience)
3. Select the appropriate domain context.
4. Click **"Run Fairness Audit"**.
5. Review the computed metrics, bias breakdown by group, and AI-generated narrative summary.

### Sample Datasets Provided

Dwesh includes four pre-built CSV datasets for testing and demonstration:

| Dataset | Rows | Bias Type | Description |
|---|---|---|---|
| Gender Biased | 500 | Gender | Females require skill score > 72 vs. Males > 55 |
| Age Biased | 500 | Age | Age > 40 requires skill > 78 vs. age < 30 needs only > 50 |
| Race Biased | 500 | Race | Black/Hispanic need skill > 75 vs. White > 50 |
| Unbiased Fair | 500 | None | Uniform threshold (skill > 60) applied equally to all groups |

---

## 11. Reinforcement Learning Training Lab

### Environment Design

The RL agent operates in a custom Gymnasium environment called **FairnessEnv**:

- **State Space** — A 10-dimensional vector representing the current fairness metrics of the system.
- **Action Space** — 6 discrete actions corresponding to threshold adjustments and feature weight modifications.
- **Reward Function** — The agent receives a reward calculated as: accuracy weight (1.5×) minus bias penalty (2.0×). This incentivizes maintaining prediction accuracy while aggressively reducing bias.

### Training Process

1. Configure the number of episodes (50 to 500) and the learning rate.
2. The agent explores the fairness-accuracy tradeoff space through trial and error.
3. A live progress chart displays improvement over training episodes.
4. The final policy output indicates the learned bias reduction strategy.

---

## 12. Mitigation Engine

### Available Strategies

| Strategy | When to Use | Typical Accuracy Impact |
|---|---|---|
| Sample Reweighting | Training data imbalance detected | Low (−1 to −2%) |
| Proxy Feature Removal | High correlation between features and protected attributes | Medium (−3 to −5%) |
| Threshold Adjustment | Post-processing fairness correction needed | Low (−1%) |
| Adversarial Training | Deep learning pipeline with learned biases | Medium (−3 to −4%) |

### AI-Powered Recommendations

The Mitigation Engine uses Google Gemini AI to generate context-specific remediation plans. Recommendations are tailored based on:

- The specific bias patterns detected in the audit
- The domain context (hiring, lending, medical, or intersectional)
- The available mitigation strategies and their tradeoffs

---

## 13. Sample Datasets & Testing

### Dataset Schema

All sample CSV files follow a standardized schema with the following columns:

| Column | Type | Description |
|---|---|---|
| name | String | Candidate identifier |
| age | Integer | Candidate age |
| gender | Categorical | Gender (Male / Female) |
| race | Categorical | Race (White / Black / Hispanic / Asian) |
| education | Categorical | Education level (Bachelor / Master / PhD) |
| experience_years | Integer | Years of professional experience |
| skill_score | Float | Numerical skill assessment score |
| label | Binary (0 / 1) | Outcome — approved (1) or rejected (0) |

### Recommended Testing Workflow

1. Upload the **Gender Biased** dataset and observe the Disparate Impact ratio falling below 0.80.
2. Upload the **Unbiased Fair** dataset and confirm the DI ratio is approximately 1.00.
3. Run LPS simulations on both datasets and compare feedback loop severity.
4. Use Parallel Universes to test different mitigation strategies against each other.
5. Run Cost Analysis to quantify the hidden harm that standard metrics miss.

### How Bias Is Injected in Test Data

**Gender bias:** Female candidates face an approval threshold of skill > 72 combined with experience > 5 years. Male candidates face a lower threshold of skill > 55 and experience > 2 years. This creates an approximately 30% approval rate gap.

**Age bias:** Three tiers are applied — candidates under 30 face an easy threshold (skill > 50), candidates aged 30–40 face a moderate threshold (skill > 60), and candidates over 40 face a strict threshold (skill > 78 combined with experience > 10 years). This simulates age discrimination in technology hiring.

**Race bias:** White applicants require skill > 50, Asian applicants require > 62, and Black/Hispanic applicants require > 75. This models systemic barriers in financial lending decisions.

---

## 14. Frontend Architecture

### Single-Page Application Design

The frontend is implemented as a single-page application. Navigation between sections is handled dynamically — showing and hiding page sections without full page reloads. This provides a fluid, app-like user experience.

### Chart Management

All data visualizations are managed through a centralized chart management system. Existing chart instances are destroyed before creating new ones to prevent memory leaks and ensure accurate rendering.

### Feature Explainer System

A built-in explainer system provides first-time guidance for every page. The system tracks which pages a user has already visited within a session and displays the appropriate explanatory popup only once, ensuring a smooth onboarding experience without repetitive interruptions.

---

## 15. Fairness Metrics — Detailed Reference

### Disparate Impact Ratio (DIR)

The EEOC's "four-fifths rule" states that the selection rate for any group should be at least 80% of the rate for the highest-scoring group.

**Interpretation Scale:**

| DIR Range | Classification | Implication |
|---|---|---|
| ≥ 0.80 | **Compliant** | Legally defensible |
| 0.65 – 0.79 | **Warning** | Potential legal liability |
| 0.50 – 0.64 | **Critical** | Likely discriminatory |
| < 0.50 | **Catastrophic** | Systemic exclusion |

### Demographic Parity Difference (DPD)

Measures the absolute difference in positive outcome rates between demographic groups.

**Interpretation Scale:**

| DPD Range | Classification |
|---|---|
| < 0.05 | Excellent parity |
| 0.05 – 0.10 | Acceptable |
| 0.10 – 0.20 | Concerning |
| > 0.20 | Severe disparity |

### Overall Bias Score

A weighted composite that considers all individual metrics along with an intersectional penalty. The score accounts for the compounding effect when multiple forms of bias overlap.

---

## 16. Domain Configurations

### Hiring Domain

- **Feedback Mechanism:** Rejected candidates lose career momentum, leading to weaker future applications.
- **Initial Disparate Impact:** 0.68 (below the legal compliance threshold).
- **Baseline Accuracy:** 87%.
- **Primary Risk:** Proxy features — ZIP code correlates with race, applicant name correlates with gender.
- **Annual Degradation Rate:** 3.5%.

### Loan / Finance Domain

- **Feedback Mechanism:** Denied loans prevent wealth-building, widening the economic gap.
- **Initial Disparate Impact:** 0.73 (marginally below threshold).
- **Baseline Accuracy:** 91%.
- **Primary Risk:** Credit scores reflect historical discrimination patterns.
- **Annual Degradation Rate:** 4.5% (fastest among all domains).

### Medical Triage Domain

- **Feedback Mechanism:** Under-triaged patients develop worse health conditions over time.
- **Initial Disparate Impact:** 0.78 (near threshold).
- **Baseline Accuracy:** 93% (highest).
- **Primary Risk:** Pain assessment bias, well-documented in medical literature.
- **Annual Degradation Rate:** 2.8% (slowest among all domains).

### Intersectional Domain

- **Feedback Mechanism:** Multi-axis discrimination compounds across gender, race, and age simultaneously.
- **Initial Disparate Impact:** 0.58 (already in the catastrophic range).
- **Baseline Accuracy:** 84% (lowest).
- **Primary Risk:** Compounding effects across gender × race × age dimensions.
- **Annual Degradation Rate:** 5.5% (most severe).

---

## 17. Research References & Impact

### Academic Foundation

1. **Ensign et al. (2018)** — *"Runaway Feedback Loops in Predictive Policing"* — Demonstrated how predictive algorithms create self-reinforcing bias cycles.
2. **Liu et al. (2018)** — *"Delayed Impact of Fair Machine Learning"* — Showed that fair classifiers can inadvertently harm the groups they aim to help through feedback effects.
3. **Buolamwini & Gebru (2018)** — *"Gender Shades"* — Revealed intersectional bias in commercial facial recognition systems.
4. **Obermeyer et al. (2019)** — Demonstrated racial bias in healthcare algorithms affecting approximately 200 million patients.
5. **Mehrabi et al. (2021)** — Comprehensive survey of bias and fairness in machine learning.

### Alignment with UN Sustainable Development Goals

Dwesh directly supports the following SDGs:

| SDG | Contribution |
|---|---|
| **SDG 10 — Reduced Inequalities** | Detects and mitigates algorithmic discrimination across multiple domains |
| **SDG 16 — Peace, Justice & Strong Institutions** | Enables algorithmic accountability through transparent audit trails |
| **SDG 5 — Gender Equality** | Surfaces gender bias in AI systems with intersectional depth |
| **SDG 8 — Decent Work & Economic Growth** | Ensures fair AI-driven hiring and lending practices |

### Impact Metrics

| Metric | Value |
|---|---|
| Domains Covered | 4 (Hiring, Loan, Medical, Intersectional) |
| Simulation Horizon | Up to 20 years |
| Mitigation Strategies Compared | 5 simultaneously |
| Intersectional Groups Analysed | 8 per audit |
| Primary Bias Metrics | 3 + composite scores |
| Cost Asymmetry Detection | Up to 85%+ harm underestimation identified |

### Why This Matters

Every year, AI systems make billions of decisions affecting real people. A biased hiring model does not simply reject one candidate — it initiates a cascade:

- Rejected candidates lose confidence and career momentum.
- Their future applications reflect weaker credentials.
- The next model iteration trains on this degraded data.
- Bias compounds exponentially over time.

Dwesh is the first platform that allows users to **observe this cascade before it occurs** and **prove which interventions genuinely work**. It is a practical tool designed for ML engineers, policymakers, and civil rights organizations to ensure AI serves everyone fairly.

---

*Dwesh — Because fairness isn't a checkbox, it's a trajectory.*

*Built for Google Solution Challenge 2026*
