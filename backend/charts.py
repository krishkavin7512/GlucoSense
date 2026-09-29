"""Every training graph as a Plotly figure plus a plain-language explanation card.

Each entry: id, title, category, figure, and explain = {what, how, read, where}
(what the graph shows, how it is computed, how to read it, where it is used in
the real world) plus a one-line insight computed from this project's numbers.
"""
import json
import math

import numpy as np
import plotly.graph_objects as go

# Palette validated for colour-blind separation on the dark card surface (#0c1122).
C1, C2, C3, C4, C5 = "#18a5a0", "#e0703a", "#8b7cf0", "#c98500", "#3f8ae6"
NEG, POS = C1, C2  # no diabetes / diabetes-or-prediabetes
INK, INK2, MUTED = "#eef2fb", "#b4bdd3", "#7c86a2"
GRID, AXIS = "#1a2238", "#2a3452"
SEQ = [[0, "#101a33"], [0.25, "#184f95"], [0.5, "#2a78d6"], [0.75, "#6da7ec"], [1, "#cde2fb"]]
DIV = [[0, "#3987e5"], [0.25, "#2c5d9c"], [0.5, "#2a2f3f"], [0.75, "#a44b4f"], [1, "#e66767"]]
FONT = "Inter, system-ui, -apple-system, Segoe UI, sans-serif"


def base_layout(**kw):
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK2, size=12),
        margin=dict(l=56, r=18, t=16, b=48),
        colorway=[C1, C2, C3, C4, C5],
        hoverlabel=dict(bgcolor="#121a30", bordercolor="#2f3a5c", font=dict(family=FONT, color=INK, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
        xaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickcolor=AXIS, ticks="outside",
                   ticklen=4, title=dict(font=dict(color=MUTED, size=12))),
        yaxis=dict(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickcolor=AXIS, ticks="outside",
                   ticklen=4, title=dict(font=dict(color=MUTED, size=12))),
        bargap=0.35,
    )
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(layout.get(k), dict):
            layout[k] = {**layout[k], **v}
        else:
            layout[k] = v
    return layout


def axis(title, **kw):
    return dict(title=dict(text=title), gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS,
                tickcolor=AXIS, ticks="outside", ticklen=4, **kw)


def to_json(fig: go.Figure) -> dict:
    return json.loads(fig.to_json())


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def graph(gid, title, category, fig, what, how, read, where, insight, height=360):
    return {"id": gid, "title": title, "category": category, "height": height,
            "figure": to_json(fig), "insight": insight,
            "explain": {"what": what, "how": how, "read": read, "where": where}}


def line(color, width=2, dash=None):
    d = dict(color=color, width=width, shape="linear")
    if dash:
        d["dash"] = dash
    return d


# --------------------------------------------------------------------------- #
def build_all(report: dict) -> list[dict]:
    S, NB, PF = report["stats"], report["nb"], report["poly"]
    G = []

    # ============================ DATA & STATISTICS ============================
    neg, pos = S["n_rows"] - S["positives"], S["positives"]
    fig = go.Figure(go.Bar(x=["No diabetes", "Diabetes / prediabetes"], y=[neg, pos],
                           marker=dict(color=[NEG, POS], cornerradius=4), width=0.45,
                           text=[f"{neg:,}", f"{pos:,}"], textposition="outside",
                           textfont=dict(color=INK), hovertemplate="%{x}: %{y:,} people<extra></extra>"))
    fig.update_layout(**base_layout(yaxis=axis("People"), showlegend=False))
    G.append(graph("class_balance", "Class balance", "Data & statistics", fig,
        "How many survey respondents fall into each class of the target variable.",
        "Count the rows where Diabetes_binary = 0 and = 1. The base rate is positives / total.",
        f"Only {pct(S['base_rate'])} of people are positive. A model that always answers \"no\" would be "
        f"{pct(1 - S['base_rate'])} accurate and completely useless, which is why this project reports recall, "
        "precision and AUC instead of accuracy.",
        "The first plot in every classification project. Imbalance this strong is normal in disease screening, "
        "credit-card fraud (well under 1% positive), equipment failure and churn prediction.",
        f"{pos:,} of {S['n_rows']:,} people ({pct(S['base_rate'])}) have diabetes or prediabetes."))

    b = S["bmi"]
    h = b["histogram"]
    edges = h["edges"]
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
    fig = go.Figure()
    for c, name, col in (("0", "No diabetes", NEG), ("1", "Diabetes / prediabetes", POS)):
        fig.add_bar(x=mids, y=h[c]["density"], name=f"{name} (histogram)", marker=dict(color=col, opacity=0.35),
                    hovertemplate="BMI %{x}: density %{y:.4f}<extra>" + name + "</extra>")
        fig.add_scatter(x=h["grid"], y=h[c]["lognormal_pdf"], name=f"{name} (log-normal fit)", mode="lines",
                        line=line(col, 2.5), hovertemplate="BMI %{x:.1f}: %{y:.4f}<extra>log-normal</extra>")
    fig.update_layout(**base_layout(barmode="overlay", bargap=0.02, xaxis=axis("Body-mass index (kg/m²)"),
                                    yaxis=axis("Probability density")))
    f0, f1 = b["fits"]["by_class"]["0"]["lognormal"], b["fits"]["by_class"]["1"]["lognormal"]
    G.append(graph("bmi_density", "BMI probability density by class", "Data & statistics", fig,
        "The probability density p(BMI | class) for each class: a normalised histogram with the fitted "
        "log-normal curve drawn over it.",
        "Histogram heights are scaled so the total area is 1. The curve is the maximum-likelihood log-normal: "
        "μ = mean(ln x), σ² = mean((ln x − μ)²), and p(x) = N(ln x | μ, σ²) / x.",
        "The orange curve sits to the right of the teal one, so higher BMI is more common among people with "
        "diabetes, but the two curves overlap heavily, so BMI alone cannot separate the classes. These two "
        "curves are exactly the class-conditional densities the Naive Bayes model multiplies together.",
        "Class-conditional densities power Naive Bayes and Gaussian discriminant analysis; density fits are "
        "used in actuarial loss models, anomaly detection (flag low-density points) and quality control.",
        f"Median BMI is {math.exp(f0['mu']):.1f} without diabetes vs {math.exp(f1['mu']):.1f} with it."))

    qq = b["qq"]
    lim = [-3.2, 3.2]
    fig = go.Figure()
    fig.add_scatter(x=lim, y=lim, mode="lines", name="Perfect normal", line=line(MUTED, 1.5, "dot"),
                    hoverinfo="skip")
    fig.add_scatter(x=qq["theoretical"], y=qq["raw_sample"], mode="markers", name="BMI",
                    marker=dict(color=C2, size=6), hovertemplate="normal %{x:.2f} → BMI %{y:.2f}<extra></extra>")
    fig.add_scatter(x=qq["theoretical"], y=qq["sample"], mode="markers", name="ln(BMI)",
                    marker=dict(color=C1, size=6), hovertemplate="normal %{x:.2f} → ln BMI %{y:.2f}<extra></extra>")
    fig.update_layout(**base_layout(xaxis=axis("Standard normal quantile"), yaxis=axis("Standardised sample quantile")))
    G.append(graph("qq_plot", "Q-Q plot: is BMI normally distributed?", "Data & statistics", fig,
        "Sample quantiles plotted against the quantiles a normal distribution would have.",
        "Sort the standardised sample, take its quantile at p = (i − 0.5)/n, and pair it with "
        "Φ⁻¹(p), the inverse normal CDF (implemented from scratch with Acklam's approximation).",
        "Points on the dotted line mean the data is normal. Raw BMI curves upward at the right: a long right "
        "tail (positive skew). ln(BMI) hugs the line far more closely, which is why the model uses a "
        "log-normal density for BMI.",
        "Checking the normality assumption behind t-tests, regression residuals and control charts; "
        "spotting fat tails in financial returns.",
        f"BMI skewness is {b['overall']['skewness']:.2f}; the log transform removes most of it."))

    xs, F = b["ecdf"]["x"], b["ecdf"]["F"]
    fig = go.Figure(go.Scatter(x=xs, y=F, mode="lines", line=dict(color=C5, width=2.5, shape="hv"),
                               name="Empirical CDF", hovertemplate="P(BMI ≤ %{x}) = %{y:.3f}<extra></extra>"))
    qs = b["quantiles"]["all"]
    for p_, lab in ((5, "P5"), (25, "Q1"), (50, "Median"), (75, "Q3"), (95, "P95")):
        v = qs[p_]
        fig.add_scatter(x=[v], y=[p_ / 100], mode="markers+text", text=[f"{lab} {v:.0f}"],
                        textposition="middle right", textfont=dict(color=INK, size=11),
                        marker=dict(color=C4, size=9, line=dict(color="#0c1122", width=2)), showlegend=False,
                        hovertemplate=f"{lab}: BMI {v:.1f}<extra></extra>")
    fig.update_layout(**base_layout(xaxis=axis("Body-mass index", range=[12, 60]),
                                    yaxis=axis("Cumulative probability", range=[0, 1.02]), showlegend=False))
    G.append(graph("ecdf", "Cumulative distribution and quantiles of BMI", "Data & statistics", fig,
        "The empirical cumulative distribution function F(t) = P(BMI ≤ t) with the main quantiles marked.",
        "Sort all 253,680 BMI values; F(t) is the fraction at or below t. The quantile Q(p) is the inverse: the "
        "value below which a fraction p of people fall (interpolated between order statistics).",
        "Read a quantile by going across from p on the y-axis to the curve and down. Half of all respondents "
        "have BMI at or below the median; the gap between Q1 and Q3 (the IQR) measures spread without being "
        "pulled around by extreme values.",
        "Child growth charts (WHO percentiles), p95/p99 response-time targets in web services, exam percentile "
        "ranks, and the 95th-percentile BMI cut-offs used in paediatrics.",
        f"Median {qs[50]:.0f}, IQR {qs[25]:.0f}–{qs[75]:.0f}, and 95% of people have BMI below {qs[95]:.0f}."))

    bc = b["by_class"]
    fig = go.Figure()
    for c, name, col in (("0", "No diabetes", NEG), ("1", "Diabetes / prediabetes", POS)):
        d = bc[c]
        lo = max(d["min"], d["q25"] - 1.5 * d["iqr"])
        hi = min(d["max"], d["q75"] + 1.5 * d["iqr"])
        fig.add_box(name=name, q1=[d["q25"]], median=[d["median"]], q3=[d["q75"]], lowerfence=[lo],
                    upperfence=[hi], mean=[d["mean"]], x=[name], marker=dict(color=col),
                    line=dict(color=col, width=2), fillcolor="rgba(0,0,0,0)", boxmean=True)
    fig.update_layout(**base_layout(yaxis=axis("Body-mass index"), showlegend=False))
    G.append(graph("box_quantiles", "Five-number summary of BMI by class", "Data & statistics", fig,
        "A box plot: minimum-ish, Q1, median, Q3 and maximum-ish BMI for each class, plus the mean (dashed).",
        "Box edges are the 25th and 75th percentiles, the line inside is the median, the dashed line is the "
        "mean. Whiskers reach 1.5 × IQR beyond the box (Tukey's rule); anything further is an outlier.",
        "The whole diabetes box is shifted upwards. The mean sits above the median in both classes, another "
        "sign of right skew.",
        "Comparing groups in clinical trials, salary bands, A/B-test results and manufacturing batches.",
        f"Mean BMI {bc['0']['mean']:.1f} vs {bc['1']['mean']:.1f}; medians {bc['0']['median']:.0f} vs {bc['1']['median']:.0f}."))

    cov = S["covariance"]
    corr = np.array(cov["corr"])
    fig = go.Figure(go.Heatmap(z=corr, x=cov["labels"], y=cov["labels"], colorscale=DIV, zmin=-1, zmax=1,
                               xgap=2, ygap=2, colorbar=dict(title=dict(text="ρ"), thickness=12, outlinewidth=0),
                               hovertemplate="%{y} × %{x}<br>correlation %{z:.3f}<extra></extra>"))
    fig.update_layout(**base_layout(margin=dict(l=170, r=10, t=10, b=150),
                                    xaxis=dict(tickangle=-45, showgrid=False, ticks=""),
                                    yaxis=dict(autorange="reversed", showgrid=False, ticks="")))
    k = cov["keys"].index("Diabetes_binary")
    top = sorted([(abs(corr[k, j]), cov["labels"][j], corr[k, j]) for j in range(len(corr)) if j != k])[-1]
    G.append(graph("correlation", "Covariance and correlation matrix", "Data & statistics", fig,
        "The Pearson correlation between every pair of the 22 columns.",
        "Covariance Cov(X, Y) = E[(X − μₓ)(Y − μᵧ)], estimated as (X − X̄)ᵀ(X − X̄)/(n − 1). Dividing by "
        "the two standard deviations gives ρ = Cov / (σₓσᵧ), which is always between −1 and 1.",
        "Red cells rise together, blue cells move in opposite directions, grey means no linear relationship. "
        "The bright block between general health, physical-health days and difficulty walking is why Naive "
        "Bayes' independence assumption is only approximate.",
        "Portfolio risk (the covariance of asset returns), multicollinearity checks before regression, sensor "
        "fusion and feature selection.",
        f"The feature most correlated with diabetes is {top[1]} (ρ = {top[2]:.2f})."))

    # ============================== PROBABILITY ===============================
    bay = S["bayes"]
    fig = go.Figure()
    labels = [r["label"] for r in bay][::-1]
    for i, r in enumerate(bay[::-1]):
        fig.add_scatter(x=[r["prior"], r["posterior"]], y=[r["label"]] * 2, mode="lines",
                        line=dict(color="#34405f", width=3), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=[r["prior"] for r in bay[::-1]], y=labels, mode="markers", name="Prior P(D)",
                    marker=dict(color=MUTED, size=9, line=dict(color="#0c1122", width=2)),
                    hovertemplate="prior %{x:.1%}<extra></extra>")
    fig.add_scatter(x=[r["posterior"] for r in bay[::-1]], y=labels, mode="markers", name="Posterior P(D | fact)",
                    marker=dict(color=POS, size=11, line=dict(color="#0c1122", width=2)),
                    customdata=[[r["likelihood_d"], r["likelihood_not_d"], r["evidence"]] for r in bay[::-1]],
                    hovertemplate="P(D | %{y}) = %{x:.1%}<br>P(fact | D) = %{customdata[0]:.1%}"
                                  "<br>P(fact | no D) = %{customdata[1]:.1%}<br>P(fact) = %{customdata[2]:.1%}<extra></extra>")
    fig.update_layout(**base_layout(xaxis=axis("Probability of diabetes / prediabetes", tickformat=".0%"),
                                    yaxis=dict(showgrid=False), margin=dict(l=190, r=18, t=16, b=48)))
    top_b = bay[0]
    G.append(graph("bayes_update", "Bayes' rule: from prior to posterior", "Probability", fig,
        "How much learning one fact about a person changes the probability that they have diabetes.",
        "Bayes' rule: P(D | F) = P(F | D) P(D) / P(F), where the evidence P(F) = P(F | D) P(D) + P(F | ¬D) P(¬D) "
        "comes from the sum and product rules. Every number is computed from counts, then cross-checked "
        "against the rate counted directly among people with the fact.",
        "The grey dot is the prior (13.9% for everyone). The orange dot is the posterior once you know the fact. "
        "The longer the bar, the more informative that fact is.",
        "Interpreting medical tests (positive predictive value), spam filtering, fault diagnosis in machines, "
        "and forensic evidence.",
        f"Knowing \"{top_b['label']}\" raises the probability from {pct(top_b['prior'])} to {pct(top_b['posterior'])}."))

    lr = sorted(bay, key=lambda r: r["likelihood_ratio"])
    fig = go.Figure(go.Bar(y=[r["label"] for r in lr], x=[r["likelihood_ratio"] for r in lr], orientation="h",
                           marker=dict(color=[POS if r["likelihood_ratio"] >= 1 else NEG for r in lr], cornerradius=4),
                           width=0.55, hovertemplate="LR+ = %{x:.2f}<extra>%{y}</extra>"))
    fig.add_vline(x=1, line=dict(color=MUTED, width=1))
    fig.update_layout(**base_layout(xaxis=axis("Likelihood ratio P(fact | D) / P(fact | no D)", type="log"),
                                    yaxis=dict(showgrid=False), showlegend=False, margin=dict(l=190, r=18, t=16, b=48)))
    G.append(graph("likelihood_ratio", "Likelihood ratios of risk factors", "Probability", fig,
        "For each fact, how many times more common it is among people with diabetes than among people without.",
        "LR = P(F | D) / P(F | ¬D). In odds form Bayes' rule is: posterior odds = LR × prior odds.",
        "LR = 1 (the grey line) means the fact carries no information. Above 1 pushes risk up, below 1 pushes it "
        "down. A log axis makes ×2 and ÷2 the same distance from the line.",
        "Evidence-based medicine rates diagnostic tests by their likelihood ratios (the Fagan nomogram); the "
        "same numbers are the weights inside a Naive Bayes classifier.",
        f"Largest LR: {lr[-1]['label']} ({lr[-1]['likelihood_ratio']:.2f}×)."))

    ti = S["target_independence"]
    ti_s = sorted(ti, key=lambda r: r["mi"])
    fig = go.Figure(go.Bar(y=[r["label"] for r in ti_s], x=[r["mi"] for r in ti_s], orientation="h",
                           marker=dict(color=C5, cornerradius=4), width=0.6,
                           customdata=[[r["chi2"], r["p_value"], r["cramers_v"]] for r in ti_s],
                           hovertemplate="I(feature; D) = %{x:.4f} bits<br>χ² = %{customdata[0]:,.0f}, "
                                         "p = %{customdata[1]:.1e}<br>Cramér's V = %{customdata[2]:.3f}<extra>%{y}</extra>"))
    fig.update_layout(**base_layout(xaxis=axis("Mutual information with diabetes (bits)"), yaxis=dict(showgrid=False),
                                    showlegend=False, margin=dict(l=190, r=18, t=16, b=48)))
    G.append(graph("independence_target", "Independence test: which features are linked to diabetes?", "Probability", fig,
        "How far each feature is from being independent of the diabetes label.",
        "Mutual information I(X; D) = Σ p(x, d) log₂[p(x, d) / (p(x) p(d))]. It is exactly 0 when "
        "p(x, d) = p(x) p(d), the definition of independence. The hover also shows the chi-square test of "
        "independence and Cramér's V.",
        "Longer bars mean stronger dependence. With 253,680 rows every p-value is essentially 0, so the "
        "size of the effect (MI or Cramér's V) matters far more than whether it is \"significant\".",
        "Feature screening before modelling, genome-wide association studies, and survey analysis.",
        f"Most informative: {ti[0]['label']} ({ti[0]['mi']:.3f} bits). Least: {ti[-1]['label']}."))

    mm = S["mi_matrix"]
    fig = go.Figure(go.Heatmap(z=mm["mi"], x=mm["labels"], y=mm["labels"], colorscale=SEQ, xgap=2, ygap=2,
                               colorbar=dict(title=dict(text="bits"), thickness=12, outlinewidth=0),
                               hovertemplate="I(%{y}; %{x}) = %{z:.4f} bits<extra></extra>"))
    fig.update_layout(**base_layout(margin=dict(l=170, r=10, t=10, b=150),
                                    xaxis=dict(tickangle=-45, showgrid=False, ticks=""),
                                    yaxis=dict(autorange="reversed", showgrid=False, ticks="")))
    G.append(graph("mi_matrix", "Pairwise dependence between features", "Probability", fig,
        "Mutual information between every pair of the 21 input features.",
        "The same MI formula applied to each pair of (binned) features: 210 joint tables computed from counts.",
        "Dark cells are close to independent. Bright cells are pairs that carry overlapping information, which "
        "Naive Bayes will double-count.",
        "Removing redundant sensors or survey questions, and building dependency graphs (Chow-Liu trees).",
        f"Most dependent pair: {S['top_dependent_pairs'][0]['a']} and {S['top_dependent_pairs'][0]['b']}."))

    pairs_all = []
    n = len(mm["keys"])
    for i in range(n):
        for j in range(i + 1, n):
            pairs_all.append((mm["mi"][i][j], mm["cmi"][i][j], f"{mm['labels'][i]} × {mm['labels'][j]}"))
    mx = max(p[0] for p in pairs_all) * 1.05
    fig = go.Figure()
    fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=line(MUTED, 1.5, "dot"), name="I(A;B|D) = I(A;B)",
                    hoverinfo="skip")
    fig.add_scatter(x=[p[0] for p in pairs_all], y=[p[1] for p in pairs_all], mode="markers", name="Feature pair",
                    text=[p[2] for p in pairs_all], marker=dict(color=C3, size=8, line=dict(color="#0c1122", width=2)),
                    hovertemplate="%{text}<br>I(A;B) = %{x:.4f}<br>I(A;B | D) = %{y:.4f}<extra></extra>")
    for p in sorted(pairs_all, key=lambda t: -t[1])[:3]:
        fig.add_annotation(x=p[0], y=p[1], text=p[2], showarrow=True, arrowcolor=MUTED, arrowwidth=1,
                           ax=-40, ay=-28, font=dict(color=INK, size=11))
    fig.update_layout(**base_layout(xaxis=axis("I(A; B)  (bits)"), yaxis=axis("I(A; B | diabetes)  (bits)")))
    G.append(graph("conditional_independence", "Conditional independence: testing the Naive Bayes assumption",
        "Probability", fig,
        "For every pair of features, dependence before and after conditioning on the diabetes label.",
        "Conditional mutual information I(A; B | D) = Σ_d P(d) I(A; B | D = d): the mutual information "
        "computed separately inside each class, then averaged.",
        "Naive Bayes assumes every point sits at y = 0 (independent given the class). Points near the diagonal "
        "stay dependent even when the class is known, so their evidence is counted twice. That double-counting is "
        "why the raw scores needed calibrating.",
        "Validating Naive Bayes and other graphical-model assumptions; causal-discovery algorithms (PC algorithm) "
        "are built entirely from conditional-independence tests.",
        f"Mean I(A;B) = {np.mean([p[0] for p in pairs_all]):.4f} bits vs mean I(A;B|D) = {np.mean([p[1] for p in pairs_all]):.4f} bits."))

    ra = S["rate_by"]["Age"]
    ramp = ["#86b6ef", "#78aded", "#6da7ec", "#5f9fea", "#5598e7", "#478fe6", "#3987e5", "#2f7fdc", "#2a78d6",
            "#2771ca", "#256abf", "#2163b5", "#1c5cab"]
    fig = go.Figure(go.Bar(x=ra["labels"], y=ra["rate"], marker=dict(color=ramp, cornerradius=4), width=0.6,
                           customdata=ra["count"],
                           hovertemplate="Age %{x}: %{y:.1%} of %{customdata:,} people<extra></extra>"))
    fig.update_layout(**base_layout(xaxis=axis("Age group"), yaxis=axis("P(diabetes | age group)", tickformat=".0%"),
                                    showlegend=False))
    G.append(graph("rate_by_age", "Conditional probability by age group", "Probability", fig,
        "P(D | age group): the share of people in each age band who have diabetes or prediabetes.",
        "Product rule rearranged: P(D | A = a) = P(D, A = a) / P(A = a), estimated by counting inside each band.",
        "Risk climbs about fifteen-fold from the 18–24 band to the early 70s, then dips slightly for 80+ (possibly because people with severe diabetes are less likely to reach that age).",
        "Age-based screening guidelines: the American Diabetes Association recommends testing everyone from age "
        "35; India's IDRS score adds points for age over 35 and 50.",
        f"{pct(ra['rate'][0])} at 18–24 vs {pct(max(ra['rate']))} at the peak."))

    rg = S["rate_by"]["GenHlth"]
    fig = go.Figure(go.Bar(x=rg["labels"], y=rg["rate"], marker=dict(color=ramp[::3][:5], cornerradius=4), width=0.5,
                           customdata=rg["count"],
                           hovertemplate="%{x}: %{y:.1%} of %{customdata:,} people<extra></extra>"))
    fig.update_layout(**base_layout(xaxis=axis("Self-rated general health"),
                                    yaxis=axis("P(diabetes | rating)", tickformat=".0%"), showlegend=False))
    G.append(graph("rate_by_health", "Conditional probability by self-rated health", "Probability", fig,
        "P(D | general-health rating) for the five answers from Excellent to Poor.",
        "Count people with diabetes inside each answer group and divide by the group size.",
        "A single, free question is a strong signal: people who rate their health as poor are many times more "
        "likely to have diabetes than people who say excellent.",
        "Self-rated health is one of the most predictive questions in population health surveys and insurance "
        "underwriting questionnaires.",
        f"{pct(rg['rate'][0])} (Excellent) vs {pct(rg['rate'][-1])} (Poor)."))

    te = S["bmi"]["total_expectation"]
    fig = go.Figure(go.Bar(x=["E[BMI | no diabetes]", "E[BMI | diabetes]", "Σ P(d) E[BMI | d]", "E[BMI] (direct)"],
                           y=[te["cond_mean"][0], te["cond_mean"][1], te["e_total"], te["e_direct"]],
                           marker=dict(color=[NEG, POS, C3, C5], cornerradius=4), width=0.5,
                           text=[f"{v:.3f}" for v in (te["cond_mean"][0], te["cond_mean"][1], te["e_total"], te["e_direct"])],
                           textposition="outside", textfont=dict(color=INK),
                           hovertemplate="%{x} = %{y:.4f}<extra></extra>"))
    fig.update_layout(**base_layout(yaxis=axis("Body-mass index", range=[20, 34]), showlegend=False))
    G.append(graph("total_expectation", "Expectation: the law of total expectation", "Probability", fig,
        "The average BMI inside each class, and how the two combine into the overall average.",
        "Law of total expectation: E[X] = Σ_d P(d) E[X | d] = 0.861 × E[BMI | no D] + 0.139 × E[BMI | D]. "
        "The same data also satisfies Var[X] = E[Var(X | D)] + Var(E[X | D]).",
        "The purple bar (built from the two conditional means) equals the blue bar (computed directly), confirming "
        "the identity numerically.",
        "Blending segment averages in business reporting, stratified sampling estimates, and the \"within-group "
        "vs between-group\" variance split behind ANOVA.",
        f"Between-class variance is only {100 * te['var_between'] / te['var_total']:.1f}% of total BMI variance."))

    # ============================== CURVE FITTING ===============================
    pts_rows = PF["degree_sweep"]["rows"]
    grid = PF["grid"]
    fig = go.Figure()
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"M = {m}" for m in (0, 1, 3, 9)],
                        horizontal_spacing=0.08, vertical_spacing=0.16)
    sample_pts = PF.get("sample_points")
    for i, M in enumerate(("0", "1", "3", "9")):
        r, c = i // 2 + 1, i % 2 + 1
        ex = PF["example_fits"][M]
        fig.add_scatter(x=sample_pts["test"][0], y=sample_pts["test"][1], mode="lines", line=line(MUTED, 1.5),
                        name="Test rate (38k people)", showlegend=i == 0, row=r, col=c,
                        hovertemplate="BMI %{x}: %{y:.1%}<extra>test</extra>")
        fig.add_scatter(x=sample_pts["train"][0], y=sample_pts["train"][1], mode="markers",
                        marker=dict(color=C5, size=7, line=dict(color="#0c1122", width=1.5)),
                        name="Training points (1,000 people)", showlegend=i == 0, row=r, col=c,
                        hovertemplate="BMI %{x}: %{y:.1%}<extra>train</extra>")
        fig.add_scatter(x=grid, y=ex["curve"], mode="lines", line=line(C2, 2.5), name="Fitted polynomial",
                        showlegend=i == 0, row=r, col=c, hovertemplate="BMI %{x:.1f}: %{y:.1%}<extra>fit</extra>")
    fig.update_layout(**base_layout(margin=dict(l=48, r=12, t=78, b=40), legend=dict(y=1.16)))
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, range=[-0.2, 0.8], tickformat=".0%")
    fig.update_annotations(font=dict(color=INK, size=12))
    G.append(graph("poly_fits", "Polynomial curve fitting for M = 0, 1, 3, 9", "Curve fitting", fig,
        "Polynomials of increasing order M fitted to the diabetes rate at each BMI, computed from a small "
        "sample of 1,000 people (the Bishop Figure 1.4 experiment on real data).",
        "Minimise E(w) = ½ Σ (y(xₙ, w) − tₙ)² with y(x, w) = Σⱼ wⱼ xʲ. Setting the gradient to zero gives the normal "
        "equations ΦᵀΦ w = Φᵀt, solved with NumPy linear algebra (x is BMI rescaled to [−1, 1]).",
        "M = 0 and 1 are too simple (underfitting). M = 3 follows the grey test curve closely. M = 9 bends to chase "
        "the noise in the sparse, extreme-BMI points (overfitting) and swings wildly between them.",
        "Calibration curves for sensors, dose–response curves in pharmacology, trajectory smoothing, and the "
        "bias–variance trade-off behind every model-complexity choice.",
        f"Test RMS error: M=0 {PF['example_fits']['0']['test_rms']:.3f}, M=3 {PF['example_fits']['3']['test_rms']:.3f}, "
        f"M=9 {PF['example_fits']['9']['test_rms']:.3f}.", height=460))

    fig = go.Figure()
    fig.add_scatter(x=[r["degree"] for r in pts_rows], y=[r["train_rms"] for r in pts_rows], mode="lines+markers",
                    name="Training", line=line(C5), marker=dict(size=8, line=dict(color="#0c1122", width=2)),
                    hovertemplate="M = %{x}: %{y:.4f}<extra>train</extra>")
    fig.add_scatter(x=[r["degree"] for r in pts_rows], y=[r["test_rms"] for r in pts_rows], mode="lines+markers",
                    name="Test", line=line(C2), marker=dict(size=8, line=dict(color="#0c1122", width=2)),
                    hovertemplate="M = %{x}: %{y:.4f}<extra>test</extra>")
    fig.update_layout(**base_layout(xaxis=axis("Polynomial order M", dtick=1), yaxis=axis("E_RMS", type="log")))
    best_m = min(pts_rows, key=lambda r: r["test_rms"])
    G.append(graph("poly_error_degree", "Training vs test error as model complexity grows", "Curve fitting", fig,
        "Root-mean-square error on the training points and on held-out test points for every order M from 0 to 12.",
        "E_RMS = √(2E(w*)/N): the typical size of the error in the same units as the target, so training and "
        "test sets of different sizes can be compared.",
        "Training error only ever goes down as M grows, because more flexible curves can always fit the same points "
        "at least as well. Test error falls, bottoms out, then explodes: past the minimum the model is memorising "
        "noise. The log axis keeps both ends readable.",
        "Choosing model complexity anywhere: tree depth, number of neurons, polynomial features, number of "
        "clusters. The U-shaped test curve is the bias–variance trade-off.",
        f"Best test error at M = {best_m['degree']} ({best_m['test_rms']:.4f}); at M = 12 it is {pts_rows[-1]['test_rms']:.1f}."))

    ls = PF["lambda_sweep"]["rows"]
    fig = go.Figure()
    fig.add_scatter(x=[r["log_lambda"] for r in ls], y=[r["train_rms"] for r in ls], mode="lines", name="Training",
                    line=line(C5), hovertemplate="ln λ = %{x:.1f}: %{y:.4f}<extra>train</extra>")
    fig.add_scatter(x=[r["log_lambda"] for r in ls], y=[r["test_rms"] for r in ls], mode="lines", name="Test",
                    line=line(C2), hovertemplate="ln λ = %{x:.1f}: %{y:.4f}<extra>test</extra>")
    fig.update_layout(**base_layout(xaxis=axis("ln λ (regularisation strength)"), yaxis=axis("E_RMS", type="log")))
    best_l = min(ls, key=lambda r: r["test_rms"])
    G.append(graph("poly_error_lambda", "Regularisation tames an M = 9 polynomial", "Curve fitting", fig,
        "Training and test error of the M = 9 polynomial as the penalty on large weights increases.",
        "Regularised error Ẽ(w) = ½ Σ (yₙ − tₙ)² + (λ/2)‖w‖², solved exactly by (ΦᵀΦ + λI) w = Φᵀt.",
        "At tiny λ (left) the M = 9 curve overfits: low training error, huge test error. As λ grows the weights "
        "shrink and test error drops. Too much λ (right) flattens the curve and both errors rise again.",
        "Weight decay in neural networks, ridge regression in econometrics, and smoothing penalties in splines.",
        f"Best test error at ln λ ≈ {best_l['log_lambda']:.1f} ({best_l['test_rms']:.4f})."))

    fig = go.Figure(go.Bar(x=[r["degree"] for r in pts_rows],
                           y=[max(abs(w) for w in r["weights"]) for r in pts_rows],
                           marker=dict(color=C3, cornerradius=4), width=0.6,
                           hovertemplate="M = %{x}: largest |w| = %{y:,.2f}<extra></extra>"))
    fig.update_layout(**base_layout(xaxis=axis("Polynomial order M", dtick=1),
                                    yaxis=axis("Largest coefficient |wⱼ|", type="log"), showlegend=False))
    G.append(graph("poly_weights", "Coefficients explode when a model overfits", "Curve fitting", fig,
        "The size of the largest polynomial coefficient for each order M (Bishop Table 1.1, as a chart).",
        "Take maxⱼ |wⱼ| from each least-squares fit. The axis is logarithmic.",
        "Low-order fits have small, sensible weights. High-order fits need enormous positive and negative "
        "weights that cancel each other exactly at the training points. That fragility is what regularisation "
        "penalises.",
        "Monitoring weight norms while training neural networks; large coefficients are a classic warning sign "
        "of multicollinearity in regression.",
        f"Largest weight grows from {max(abs(w) for w in pts_rows[1]['weights']):.2f} (M=1) to "
        f"{max(abs(w) for w in pts_rows[-1]['weights']):,.0f} (M=12)."))

    fig = go.Figure()
    fig.add_scatter(x=sample_pts["test"][0], y=sample_pts["test"][1], mode="lines", line=line(MUTED, 1.5),
                    name="Test rate", hovertemplate="BMI %{x}: %{y:.1%}<extra>test</extra>")
    for s, col, name in (("300", C4, "300 people"), ("3000", C3, "3,000 people"), ("all", C1, "All 177,576")):
        e = PF["size_effect"][s]
        fig.add_scatter(x=grid, y=e["curve"], mode="lines", line=line(col, 2.5), name=f"M = 9, {name}",
                        hovertemplate="BMI %{x:.1f}: %{y:.1%}<extra>" + name + "</extra>")
    fig.update_layout(**base_layout(xaxis=axis("Body-mass index"),
                                    yaxis=axis("Fraction with diabetes", range=[-0.2, 0.8], tickformat=".0%")))
    G.append(graph("poly_data_size", "More data cures overfitting", "Curve fitting", fig,
        "The same M = 9 polynomial fitted to training sets of different sizes (Bishop Figure 1.6).",
        "Identical least-squares fit; only the number of people behind the training points changes.",
        "With 300 people the M = 9 curve is wild. With the full training set the same flexible model settles onto "
        "the true curve, because the noise in each point averages out.",
        "Why big-data companies can train huge models, and why small medical studies must use simple ones.",
        " · ".join(f"{k}: test RMS {v['test_rms']:.3f}" for k, v in PF["size_effect"].items())))

    sel = PF["selected"]
    allp = PF["all_points"]
    fig = go.Figure()
    fig.add_scatter(x=allp[0], y=allp[1], mode="markers", name="Training rate at each BMI",
                    marker=dict(color=C5, size=[max(5, min(16, math.sqrt(c) / 6)) for c in allp[2]],
                                line=dict(color="#0c1122", width=1.5)),
                    customdata=allp[2], hovertemplate="BMI %{x}: %{y:.1%} of %{customdata:,}<extra></extra>")
    fig.add_scatter(x=grid, y=sel["curve"], mode="lines", line=line(C2, 3), name=f"Selected fit (M = {sel['degree']})",
                    hovertemplate="BMI %{x:.1f}: %{y:.1%}<extra>fit</extra>")
    fig.update_layout(**base_layout(xaxis=axis("Body-mass index"),
                                    yaxis=axis("Fraction with diabetes", tickformat=".0%")))
    G.append(graph("poly_selected", "The trained population risk curve", "Curve fitting", fig,
        "The final polynomial, chosen on the validation set, that turns a BMI into the expected share of people "
        "with diabetes. The screener shows this value next to each personal prediction.",
        "Every order M from 0 to 10 and several λ values were fitted to the training rates; the pair with the "
        "lowest validation E_RMS was kept, then scored once on the untouched test set.",
        "Dot size shows how many people sit at each BMI. The curve rises steeply through the overweight range and "
        "flattens above BMI 45, where the data thins out.",
        "Population risk tables in public-health reports and actuarial pricing curves.",
        f"M = {sel['degree']}, validation RMS {sel['val_rms']:.4f}, test RMS {sel['test_rms']:.4f}."))

    # ================================ TRAINING ================================
    lc = NB["learning_curve"]
    fig = go.Figure()
    fig.add_scatter(x=[r["size"] for r in lc], y=[r["train_auc"] for r in lc], mode="lines+markers", name="Training",
                    line=line(C5), marker=dict(size=8, line=dict(color="#0c1122", width=2)),
                    hovertemplate="%{x:,} rows: AUC %{y:.4f}<extra>train</extra>")
    fig.add_scatter(x=[r["size"] for r in lc], y=[r["val_auc"] for r in lc], mode="lines+markers", name="Validation",
                    line=line(C2), marker=dict(size=8, line=dict(color="#0c1122", width=2)),
                    hovertemplate="%{x:,} rows: AUC %{y:.4f}<extra>validation</extra>")
    fig.update_layout(**base_layout(xaxis=axis("Training rows", type="log"), yaxis=axis("ROC AUC")))
    G.append(graph("learning_curve", "Learning curve", "Training", fig,
        "Model quality on the rows it trained on vs on unseen validation rows, as the training set grows.",
        "Train Naive Bayes on random subsets of 200 to 177,576 rows (averaged over 3 draws) and score both sets.",
        "A big gap between the lines means overfitting (high variance). Lines that meet and flatten mean more "
        "data will not help much; a better model would. Naive Bayes has few parameters, so it converges fast.",
        "Deciding whether to spend money collecting more labelled data or on a better model.",
        f"Validation AUC {lc[0]['val_auc']:.3f} with 200 rows, {lc[-1]['val_auc']:.3f} with all rows."))

    sw = NB["alpha_sweep"]
    fig = go.Figure()
    for key, name, col in (("full", "Trained on 177,576 rows", C1), ("small", "Trained on 1,000 rows", C2)):
        fig.add_scatter(x=[r["alpha"] for r in sw[key]], y=[r["auc"] for r in sw[key]], mode="lines+markers", name=name,
                        line=line(col), marker=dict(size=8, line=dict(color="#0c1122", width=2)),
                        hovertemplate="α = %{x}: AUC %{y:.4f}<extra>" + name + "</extra>")
    fig.add_vline(x=NB["alpha"], line=dict(color=MUTED, width=1))
    fig.update_layout(**base_layout(xaxis=axis("Laplace smoothing α", type="log"), yaxis=axis("Validation ROC AUC")))
    G.append(graph("alpha_validation", "Validation curve for Laplace smoothing", "Training", fig,
        "How the smoothing hyperparameter α affects validation AUC, for a large and a small training set.",
        "Categorical likelihoods are estimated as (count + α) / (N_class + Kα). α acts as imaginary extra "
        "observations of every answer, which stops rare answers from getting probability 0.",
        "With lots of data, α barely matters until it is huge. With 1,000 rows, too much smoothing drowns the real "
        "signal quickly. The grey line marks the chosen α.",
        "Hyperparameter tuning in any model: smoothing in spam filters and language models, regularisation strength, "
        "learning rates.",
        f"Chosen α = {NB['alpha']} (Laplace's rule); every α ≤ 3 scores within 0.0001 AUC."))

    cv = NB["cv"]
    mean_cv = float(np.mean([c["auc"] for c in cv]))
    fig = go.Figure(go.Bar(x=[f"Fold {c['fold']}" for c in cv], y=[c["auc"] for c in cv],
                           marker=dict(color=C5, cornerradius=4), width=0.45,
                           hovertemplate="%{x}: AUC %{y:.4f}<extra></extra>"))
    fig.add_hline(y=mean_cv, line=dict(color=C2, width=2), annotation_text=f"mean {mean_cv:.4f}",
                  annotation_font_color=INK, annotation_position="top left")
    fig.update_layout(**base_layout(yaxis=axis("ROC AUC", range=[min(c["auc"] for c in cv) - 0.01,
                                                                 max(c["auc"] for c in cv) + 0.01]), showlegend=False))
    G.append(graph("cross_validation", "5-fold cross-validation", "Training", fig,
        "The model's AUC on each of five folds, each fold held out once while the other four train the model.",
        "Split the training + validation rows into 5 stratified folds (every fold keeps the 13.9% base rate), "
        "train 5 models, and score each on its held-out fold.",
        "Bars of similar height mean the result does not depend on a lucky split. The spread is the uncertainty in "
        "the reported score.",
        "Standard practice for reporting model performance in research papers and Kaggle competitions, and for "
        "tuning hyperparameters without touching the test set.",
        f"AUC {mean_cv:.4f} ± {np.std([c['auc'] for c in cv]):.4f} across folds."))

    llr = sorted(NB["llr"], key=lambda r: abs(r["llr"]))[-22:]
    fig = go.Figure(go.Bar(y=[f"{r['feature']}: {r['level']}" for r in llr], x=[r["llr"] for r in llr], orientation="h",
                           marker=dict(color=[POS if r["llr"] > 0 else NEG for r in llr], cornerradius=4), width=0.62,
                           hovertemplate="log LR = %{x:+.3f} (×%{customdata:.2f})<extra>%{y}</extra>",
                           customdata=[math.exp(r["llr"]) for r in llr]))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(**base_layout(xaxis=axis("log[ p(answer | D) / p(answer | no D) ]"), yaxis=dict(showgrid=False),
                                    showlegend=False, margin=dict(l=240, r=18, t=16, b=48)))
    G.append(graph("nb_weights", "What the Naive Bayes model learned", "Training", fig,
        "The 22 answers that move the prediction most, as log-likelihood ratios learned from the training data.",
        "For every answer level, log p(x = k | D) − log p(x = k | ¬D). A person's posterior log-odds is the prior "
        "log-odds plus the sum of these terms for their answers.",
        "Orange bars push towards diabetes, teal bars push away. A value of +0.69 doubles the odds; −0.69 halves them.",
        "Explaining a model's decision to a doctor or a customer: the same additive evidence scores are used in "
        "credit scorecards and clinical risk scores.",
        f"Strongest single answer: {llr[-1]['feature']} = {llr[-1]['level']} (×{math.exp(llr[-1]['llr']):.2f} odds)."))

    # =============================== EVALUATION ===============================
    roc = NB["roc"]
    sk = NB["sklearn"]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=line(MUTED, 1.5, "dot"), name="Random guessing (0.5)",
                    hoverinfo="skip")
    for key, name, col, auc_ in (("sklearn_gaussian", "sklearn GaussianNB", C3, sk["auc_sklearn_gaussian"]),
                                 ("sklearn_categorical", "sklearn CategoricalNB", C4, sk["auc_sklearn_categorical"]),
                                 ("ours", "Ours: mixed Naive Bayes", C2, sk["auc_ours_mixed"])):
        fig.add_scatter(x=roc[key]["fpr"], y=roc[key]["tpr"], mode="lines", line=line(col, 2.5),
                        name=f"{name} ({auc_:.3f})",
                        hovertemplate="FPR %{x:.3f}, TPR %{y:.3f}<extra>" + name + "</extra>")
    t = NB["test"]
    fig.add_scatter(x=[1 - t["specificity"]], y=[t["recall"]], mode="markers", name="Chosen screening point",
                    marker=dict(color=INK, size=11, symbol="diamond", line=dict(color="#0c1122", width=2)),
                    hovertemplate="recall %{y:.1%}, false-alarm rate %{x:.1%}<extra></extra>")
    fig.update_layout(**base_layout(xaxis=axis("False-positive rate (1 − specificity)", range=[0, 1]),
                                    yaxis=axis("True-positive rate (recall)", range=[0, 1.01])))
    G.append(graph("roc", "ROC curve", "Evaluation", fig,
        "The trade-off between catching real cases and raising false alarms, over every possible threshold, on "
        "the 38,052-person test set.",
        "Sort people by predicted risk; at each cut-off compute TPR = TP/(TP+FN) and FPR = FP/(FP+TN). The area "
        "under the curve (AUC, by the trapezoid rule) is the probability that a random positive person is ranked "
        "above a random negative one.",
        "Closer to the top-left corner is better; the dotted diagonal is a coin flip. The diamond is the "
        "screening point the app uses (about 80% recall).",
        "Comparing diagnostic tests in medicine, credit-score models, radar detection (where ROC analysis began) "
        "and biometric systems.",
        f"Our AUC {sk['auc_ours_mixed']:.3f} beats sklearn GaussianNB ({sk['auc_sklearn_gaussian']:.3f}) by modelling each "
        "feature with a density that suits it."))

    pr = NB["pr"]["ours"]
    fig = go.Figure()
    fig.add_scatter(x=pr["recall"], y=pr["precision"], mode="lines", line=line(C2, 2.5), name="Mixed Naive Bayes",
                    hovertemplate="recall %{x:.3f}, precision %{y:.3f}<extra></extra>")
    fig.add_hline(y=t["base_rate"], line=dict(color=MUTED, width=1.5, dash="dot"),
                  annotation_text=f"base rate {pct(t['base_rate'])}", annotation_font_color=INK2,
                  annotation_position="bottom right")
    fig.add_scatter(x=[t["recall"]], y=[t["precision"]], mode="markers", name="Chosen screening point",
                    marker=dict(color=INK, size=11, symbol="diamond", line=dict(color="#0c1122", width=2)),
                    hovertemplate="recall %{x:.1%}, precision %{y:.1%}<extra></extra>")
    fig.update_layout(**base_layout(xaxis=axis("Recall (share of real cases caught)", range=[0, 1]),
                                    yaxis=axis("Precision (share of flags that are real)", range=[0, 1])))
    G.append(graph("pr_curve", "Precision–recall curve", "Evaluation", fig,
        "For every threshold, what fraction of flagged people truly have diabetes (precision) vs what fraction of "
        "all real cases get flagged (recall).",
        "Precision = TP/(TP+FP), recall = TP/(TP+FN), computed at every distinct score. Average precision "
        "summarises the curve as Σ (Rₖ − Rₖ₋₁) Pₖ.",
        "With a rare positive class this is more honest than ROC: random guessing only achieves the base rate "
        "(dotted line). Pushing recall up always costs precision.",
        "Fraud detection, information retrieval and search ranking, and rare-disease screening.",
        f"At {pct(t['recall'])} recall, {pct(t['precision'])} of flagged people truly are positive "
        f"(2× the base rate); average precision {t['average_precision']:.3f}."))

    tc = NB["threshold_curve"]
    fig = go.Figure()
    for key, name, col in (("recall", "Recall", C2), ("precision", "Precision", C1), ("specificity", "Specificity", C5),
                           ("flagged_rate", "Share of people flagged", C4)):
        fig.add_scatter(x=tc["threshold"], y=tc[key], mode="lines", line=line(col, 2.5), name=name,
                        hovertemplate="threshold %{x:.1%}: %{y:.1%}<extra>" + name + "</extra>")
    fig.add_vline(x=NB["threshold_calibrated"], line=dict(color=INK, width=1),
                  annotation_text="app default", annotation_font_color=INK, annotation_position="top right")
    fig.update_layout(**base_layout(xaxis=axis("Decision threshold on calibrated risk", tickformat=".0%"),
                                    yaxis=axis("Rate", tickformat=".0%", range=[0, 1.02])))
    G.append(graph("threshold_tradeoff", "Choosing the decision threshold", "Evaluation", fig,
        "How recall, precision, specificity and the share of people sent for a blood test change with the "
        "threshold.",
        "Sweep the threshold from 0.5% to 50% calibrated risk and recompute the confusion matrix at each step.",
        "Moving right flags fewer people: precision and specificity rise but recall collapses. A screener "
        "deliberately sits on the left side, because a missed case costs far more than an extra blood test.",
        "Setting alert thresholds in fraud systems, cancer screening programmes and spam filters; the sensitivity "
        "slider in this app moves along exactly this curve.",
        f"Default threshold {pct(NB['threshold_calibrated'])}: recall {pct(t['recall'])}, "
        f"{pct(t['flagged_rate'])} of people flagged."))

    cm = [[t["tn"], t["fp"]], [t["fn"], t["tp"]]]
    tot = sum(sum(r) for r in cm)
    txt = [[f"{v:,}<br>{v / tot:.1%}" for v in row] for row in cm]
    fig = go.Figure(go.Heatmap(z=cm, x=["Predicted: no", "Predicted: yes (test)"], y=["Actual: no", "Actual: yes"],
                               colorscale=SEQ, showscale=False, xgap=3, ygap=3, text=txt, texttemplate="%{text}",
                               textfont=dict(size=15, color=INK),
                               hovertemplate="%{y}, %{x}: %{z:,}<extra></extra>"))
    fig.update_layout(**base_layout(xaxis=dict(showgrid=False, ticks="", side="bottom"),
                                    yaxis=dict(showgrid=False, ticks="", autorange="reversed"),
                                    margin=dict(l=100, r=10, t=10, b=48)))
    G.append(graph("confusion_matrix", "Confusion matrix at the screening threshold", "Evaluation", fig,
        "The four outcomes on the test set: true negatives, false alarms, missed cases and caught cases.",
        "Compare each person's thresholded prediction with their true label and count each combination.",
        "Bottom-right = caught cases (TP), bottom-left = missed cases (FN), top-right = false alarms (FP). "
        "Recall = TP/(TP+FN); precision = TP/(TP+FP).",
        "Every classifier report. In medicine, FN and FP carry very different costs, so the matrix drives the "
        "threshold choice.",
        f"{t['tp']:,} cases caught, {t['fn']:,} missed, {t['fp']:,} extra blood tests."))

    cal = NB["calibration"]
    fig = go.Figure()
    fig.add_scatter(x=[0, 0.7], y=[0, 0.7], mode="lines", line=line(MUTED, 1.5, "dot"), name="Perfect calibration",
                    hoverinfo="skip")
    fig.add_scatter(x=cal["raw"]["mean_predicted"], y=cal["raw"]["observed"], mode="lines+markers",
                    name="Raw Naive Bayes", line=line(C3), marker=dict(size=8, line=dict(color="#0c1122", width=2)),
                    hovertemplate="predicted %{x:.1%} → observed %{y:.1%}<extra>raw</extra>")
    fig.add_scatter(x=cal["calibrated"]["mean_predicted"], y=cal["calibrated"]["observed"], mode="lines+markers",
                    name="After calibration", line=line(C2), marker=dict(size=8, line=dict(color="#0c1122", width=2)),
                    hovertemplate="predicted %{x:.1%} → observed %{y:.1%}<extra>calibrated</extra>")
    fig.update_layout(**base_layout(xaxis=axis("Predicted probability", tickformat=".0%", range=[0, 1]),
                                    yaxis=axis("Observed frequency", tickformat=".0%", range=[0, 0.7])))
    G.append(graph("calibration", "Reliability diagram (calibration)", "Evaluation", fig,
        "Whether a predicted \"30% risk\" really means 30 out of 100 such people have diabetes.",
        "Sort test people by prediction, cut them into 12 equal-count bins, and plot the mean prediction against "
        "the observed positive rate in each bin. Calibration maps raw scores to P(D | score bin) counted on the "
        "validation set, with Laplace smoothing and a monotone constraint.",
        "Points on the diagonal are honest probabilities. Raw Naive Bayes is badly overconfident (it says 96% "
        "where the truth is 45%), because correlated answers get counted as independent evidence. After "
        "calibration the points sit on the line.",
        "Weather forecasting (\"70% chance of rain\" must verify 70% of the time), clinical risk calculators and "
        "insurance pricing all require calibrated probabilities.",
        f"Test log-loss improves from {t['log_loss']:.3f} to {t['log_loss_calibrated']:.3f}; Brier score "
        f"{t['brier']:.3f} → {t['brier_calibrated']:.3f}."))

    cb = NB["calibrator"]
    raw_p = [1 / (1 + math.exp(-x)) for x in cb["x"]]
    fig = go.Figure()
    fig.add_scatter(x=raw_p, y=cb["raw_rates"], mode="markers", name="Observed rate in each validation bin",
                    marker=dict(color=C5, size=8, line=dict(color="#0c1122", width=2)),
                    hovertemplate="raw %{x:.2%} → observed %{y:.1%}<extra></extra>")
    fig.add_scatter(x=raw_p, y=cb["y"], mode="lines", line=dict(color=C2, width=2.5, shape="linear"),
                    name="Monotone calibration map", hovertemplate="raw %{x:.2%} → %{y:.1%}<extra></extra>")
    fig.update_layout(**base_layout(xaxis=axis("Raw Naive Bayes score", type="log", tickformat=".1%"),
                                    yaxis=axis("Calibrated probability", tickformat=".0%")))
    G.append(graph("calibration_map", "The calibration function", "Evaluation", fig,
        "The function that converts a raw Naive Bayes score into a calibrated probability.",
        "30 equal-count bins of validation people; in each bin P(D | bin) = (positives + 1)/(people + 2) (Laplace's "
        "rule of succession). Bins that break the ordering are merged (pool-adjacent-violators), then the map is "
        "interpolated on the log-odds scale.",
        "The map is flatter than the diagonal: it pulls extreme scores back towards realistic values.",
        "Probability calibration layers sit on top of production classifiers at banks, ad platforms "
        "(click-through prediction) and hospitals.",
        f"A raw score of {pct(raw_p[-1], 0)} maps to {pct(cb['y'][-1])}."))

    sh = NB["score_hist"]
    e = sh["edges"]
    mids = [(e[i] + e[i + 1]) / 2 for i in range(len(e) - 1)]
    nneg, npos = sum(sh["neg"]), sum(sh["pos"])
    fig = go.Figure()
    fig.add_bar(x=mids, y=[v / nneg for v in sh["neg"]], name="No diabetes", marker=dict(color=NEG, opacity=0.6),
                hovertemplate="risk %{x:.1%}: %{y:.1%} of negatives<extra></extra>")
    fig.add_bar(x=mids, y=[v / npos for v in sh["pos"]], name="Diabetes / prediabetes", marker=dict(color=POS, opacity=0.6),
                hovertemplate="risk %{x:.1%}: %{y:.1%} of positives<extra></extra>")
    fig.add_vline(x=NB["threshold_calibrated"], line=dict(color=INK, width=1),
                  annotation_text="threshold", annotation_font_color=INK)
    fig.update_layout(**base_layout(barmode="overlay", bargap=0.04, xaxis=axis("Calibrated risk", tickformat=".0%"),
                                    yaxis=axis("Share of class", tickformat=".0%")))
    G.append(graph("score_distribution", "Predicted-risk distribution by true class", "Evaluation", fig,
        "Histograms of the predicted risk, drawn separately for people who do and do not have diabetes.",
        "Bin the calibrated predictions of the test set and normalise each class to 100%.",
        "Good separation means the two humps sit apart. Everything right of the threshold gets flagged; the teal "
        "area there is false alarms, and the orange area to the left is missed cases.",
        "Setting cut-offs in credit scoring (good vs bad borrower score distributions) and in anomaly detection.",
        f"Most people without diabetes score below {pct(NB['threshold_calibrated'])}."))

    ag = sk["agreement_sample"]
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=line(MUTED, 1.5, "dot"), name="Identical",
                    hoverinfo="skip")
    fig.add_scatter(x=ag["sklearn"], y=ag["ours"], mode="markers", name="Test respondent",
                    marker=dict(color=C1, size=7, opacity=0.8),
                    hovertemplate="sklearn %{x:.6f}<br>ours %{y:.6f}<extra></extra>")
    fig.update_layout(**base_layout(xaxis=axis("scikit-learn CategoricalNB probability"),
                                    yaxis=axis("Our from-scratch probability")))
    G.append(graph("sklearn_check", "From-scratch vs scikit-learn", "Evaluation", fig,
        "Proof that the hand-written Naive Bayes computes exactly what the reference library computes.",
        "Fit our model and sklearn's CategoricalNB on the same all-categorical data with the same α, then compare "
        "the predicted probabilities for 1,500 test people.",
        "Every point lies on the diagonal: the two implementations agree to floating-point precision.",
        "Unit-testing ML code against reference implementations is standard practice before deploying a model.",
        f"Largest difference over the whole test set: {sk['max_abs_diff_categorical']:.1e}."))
    return G
