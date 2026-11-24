# Literature notes: Linear ML slice (Victor)

One line per paper, in my own words: what they found, and which of our choices it backs.
These become the sentences for the sections 4.3 and 5.3 of the report.

Terms used throughout:
- HAR: the standard volatility benchmark; a linear regression of future volatility on its daily, weekly and monthly averages.
- Realised volatility: volatility measured from observed returns.
- Diebold–Mariano test: tests whether two models' forecast errors differ significantly.

## Section 4.3(a) Machine learning can match or beat HAR
Christensen, Siggaard & Veliyev (2023). *J. Financial Econometrics* 21(5), 1680–1727. Cite the journal version.

### What the authors did
- Compared machine-learning (ML) models, including Elastic Net (EN), with HAR models on US stocks.
    - 29 Dow Jones (DJIA) stocks, Jan 2001 – Dec 2017; realised volatility (RV) from 5-minute returns; models compared by mean squared error (MSE)
    - Diebold–Mariano tests (DM) between every pair of models (row = benchmark, column = model; Tables 2–7)
    - Machine-learning models tuned minimally
- Used two sets of predictors.
    - Lags-only set (MHAR): only the daily, weekly and monthly lags of realised volatility
    - Full set (MALL): the lags plus 9 extra predictors, including the VIX and model-free implied volatility (IV)
    - Footnote 16: the authors also ran implied volatility replaced by the residual of implied volatility regressed on realised volatility; they report "roughly unchanged" results

### What they found
- Machine learning beats HAR, even with only the lags (abstract); but with only the lags, it's mainly the neural networks (NN). Source: Section 3, Table 2, one day.
    - The authors call the lags-only results "mixed"
    - Neural networks: about 5% lower mean squared error than HAR on average; Diebold–Mariano test frequently rejected at 10% or 5%
    - Regularised linear models: "at par" with HAR
    - Tree models: worse than HAR; random forest (RF) about equal
    - Section 4: a longer horizon doesn't change the lags-only conclusions; neural networks "still dominating"
- Machine learning's gains grow with the horizon, but only with the full set (Section 4).
    - Lags-only set: a longer horizon "does not alter the conclusions"
    - Figure 8: random forest and neural networks look more persistent than HAR only at one month; at one day, about equal
- Adding predictors to HAR without a penalty doesn't help (Section 3).
    - Full set, one day: HAR with the extra predictors has only 3.4% lower mean squared error than HAR; three other HAR extensions do worse than HAR
    - The authors attribute this to in-sample overfitting

### Elastic Net results
- One day ahead: Elastic Net beats HAR once the extra predictors are added (Section 3, Table 3, full set).
    - Mean squared error relative to HAR: Elastic Net 0.916 (8.4% lower), ridge (RR) 0.919, random forest 0.901, HAR on logs (LogHAR) 0.901
    - The authors: the random forest–Elastic Net gap is "marginal" and typically not significant
    - The authors call Elastic Net "on top of the linear predictions"; they class HAR on logs as nonlinear
    - Lags-only set: regularisation only "at par" with HAR; the authors write it is "more potent when growing the feature space"
- One week ahead (5 trading days): Elastic Net still beats HAR; random forest beats Elastic Net (Table 5, full set).
    - Elastic Net vs HAR: 0.925 (7.5% lower mean squared error), significant at 5%
    - Elastic Net vs HAR on logs: 0.958, not significant
    - Random forest vs Elastic Net: 0.879 (12.1% lower), significant at 1%
    - Bagged trees (BG) vs Elastic Net: 0.928, significant at 5%; gradient boosting (GB) vs Elastic Net: 1.022, not significant
    - "Significant" here: the one-sided Diebold–Mariano test rejects equal error for more than 50% of the stocks (Table 5 notes; PDF p. 1706)

### Links to our design (authors' facts, ours next to them)
- Target: average realised volatility over the next week (days t+1 to t+5), using information up to day t (Section 4).
    - Ours: log of SPY realised volatility over the next 5 trading days
- Scaling: inputs standardised with the training set's mean and variance before ridge, lasso (LA), Elastic Net and the neural networks (Section 2; footnote 3).
    - Ours: StandardScaler inside the Pipeline
- VIX: one of the full-set predictors, in levels; only HAR on logs takes its log (Section 2).
    - Ours: f4 = log VIX
- SPY: the authors report, without numbers, that a rerun on SPY (and on the Dow Jones index) confirmed their findings (Section 2, footnote 12).

### Caveats for the Q&A
- Their realised volatility comes from 5-minute intraday returns (Section 2); ours from daily data.
- Individual stocks, not an index fund; only footnote 12 covers SPY.
- Their Elastic Net mixing parameter runs the other way: a = 0 is lasso, a = 1 is ridge (footnote 2).
    - Outside the material (scikit-learn docs): l1_ratio = 1 is lasso, l1_ratio = 0 is ridge
- "Why LightGBM and not random forest?"
    - The paper doesn't back LightGBM specifically. Project Plan v4 (part 2) cut the project from six models to four as too much to defend, without a reason per model; random forest was in earlier plans
    - One week ahead, full set (Table 5): random forest vs HAR 0.796 (significant at 1%); gradient boosting (LightGBM's family) vs HAR 0.930 (significant at 10%)
    - Gradient boosting vs Elastic Net: 1.022, not significant
    - Answer: both linear and tree models beat HAR with extra predictors, so we test one of each; gradient boosting beat HAR too, but less strongly than random forest

## Section 4.3(b) The VIX adds information beyond past realised volatility
Blair, Poon & Taylor (2001). Forecasting S&P 100 volatility: the incremental information content of implied
volatilities and high-frequency index returns. *J. Econometrics* 105(1), 5–26.

### What the authors did
- Compared the VIX with forecasts built from past returns, for S&P 100 volatility (Sections 2–3).
    - Daily data, 1987–1999; fitted on 1987–1992, tested on 4 Jan 1993 – 31 Dec 1999 (1,768 days)
    - Forecasts made every day for the next 1, 5, 10 and 20 days (Section 3.2)
    - GARCH-type models, re-estimated daily on a rolling window of the last 1,000 days
- Forecast two measures of realised volatility.
    - Sum of squared daily excess returns over the next N days
    - Sum of intraday volatility (INTRA; squared 5-minute returns plus the overnight return) over the next N days
- Four competing forecasts: historic volatility (HV; variance of the last 100 daily returns), a GARCH model (GJR), intraday volatility, the VIX.
    - Their VIX: the old S&P 100 index, an at-the-money option with 22 trading days to expiry (Section 2.2)

### What they found
- The VIX gives the most accurate forecasts at every horizon (abstract; Section 4).
    - The VIX beats the other single forecasts in all 16 combinations of measure, horizon and accuracy statistic
    - Footnote 6: few of the differences in accuracy are statistically significant at conventional levels
- At 1 and 5 days, adding the GARCH forecast to the VIX adds a little; at 10 and 20 days, it adds nothing (Section 4, Table 3).
- Intraday returns add little beyond the VIX (abstract; Section 5).

### Results at our horizon (5 days, Tables 2–3)
- Share of variability explained, P, intraday measure (Table 2B); higher P = lower mean squared error (MSE):
    - VIX 0.533; intraday volatility 0.494; GARCH 0.289; historic volatility 0.243
- Same, squared-returns measure (Table 2A):
    - VIX 0.239; intraday volatility 0.204; historic volatility 0.089; GARCH 0.085
- R² from regressing realised volatility on the forecasts, intraday measure (Table 3B):
    - VIX 0.567; VIX and GARCH together 0.586; GARCH alone 0.449

### Links to our design (authors' facts, ours next to them)
- Horizon: 5-day forecasts made every day (Section 3.2).
    - Ours: log realised volatility over the next 5 trading days, one row per day
- Implied volatility: the S&P 100 VIX, in levels, inside the GARCH variance equation (Eq. 7).
    - Ours: f4 = log of today's S&P 500 VIX (FRED)
- Benchmarks: historic volatility, GARCH and intraday volatility, not HAR.
    - Ours: HAR

### Caveats for the Q&A
- Different index (S&P 100), different VIX (the old S&P 100 version) and different years (1987–1999).
- Their benchmarks are not HAR.

### Why not Busch, Christensen & Nielsen (2011)
- busch2011.pdf: the forecast target is realised volatility over the next 22 trading days, sampled at t = 22, 44, … (Eq. 14).
    - Daily and weekly realised volatility appear only as past values on the right-hand side
    - The authors forecast one-month volatility because the options expire monthly

## Section 4.3(c) Stress variables carry information about future volatility
Christiansen, Schmeling & Schrimpf (2012). A comprehensive look at financial volatility prediction by economic
variables. *J. Applied Econometrics* 27(6), 956–977.

### What the authors did
- Tested whether macro and financial variables predict volatility in four asset classes (Sections 1–2).
    - Stocks, currencies, bonds, commodities
    - Long sample (US stocks): Dec 1926 – Dec 2010, 1,009 months
    - Short sample (all four classes): Jan 1983 – Dec 2010, 366 months
- Measured volatility monthly (Section 2.1).
    - Volatility = log of the square root of the sum of squared daily returns in the month
- Forecast one month ahead (footnote 11).
    - Longer horizons (quarterly data) are in the online appendix; the authors say they add little
- Chose predictors by Bayesian model averaging (BMA) and by classical model selection, from up to 38 candidates (Sections 1 and 3).
    - Benchmark: last month's volatility only (a one-lag autoregressive model, AR(1)); three lags, AR(3), as a check
- Credit and funding-stress predictors (Section 2.2):
    - Default spread (DEF): yield gap between BAA- and AAA-rated corporate bonds (credit risk)
    - TED spread: 3-month LIBOR minus the T-bill rate (funding stress between banks)

### What they found
- Credit-risk and funding-stress measures predict volatility across asset classes (abstract; Section 5).
    - TED spread: among the best predictors for currency, bond and stock volatility; a higher spread → higher future volatility (Section 4.1)
    - Default spread: predicts stock and bond volatility (Section 4.1)
    - The predictive power holds after controlling for past volatility (Section 5)
- Purely macroeconomic variables add little (abstract; Section 5).
- Out-of-sample gains vary by asset class and over time (abstract; Section 4.3).
    - Long US stock sample: models with the extra predictors generally beat the one-lag benchmark
    - Short sample: "less favorable" than the long stock sample; less success against the three-lag benchmark

### Results for stocks, short sample (Table IX, vs the one-lag benchmark, forecasts from Feb 1993)
- Theil's U (forecast error relative to the benchmark; below 1 = better): 0.960 to 1.010 across the six forecasting methods
- Out-of-sample R²: 0.004 to 0.078
- Bootstrap p-value: 0.00 for four methods, 0.06 for one, 0.49 for the one with Theil's U 1.010

### Links to our design (authors' facts, ours next to them)
- Volatility measure: log of the square root of realised variance from daily returns (Section 2.1).
    - Ours: y = log of SPY realised volatility from daily data
- Stress predictors: default spread and TED spread (Section 2.2).
    - Ours: f6 = TLT (long-term Treasury bond fund) volatility, f7 = bank-sector vs market volatility (XLF vs SPY)
    - The authors don't test TLT or XLF measures
- Horizon: one month (footnote 11).
    - Ours: 5 trading days

### Caveats for the Q&A
- Monthly data and a one-month horizon, not 5 days.
- Their stress variables are interest-rate spreads, not market volatilities.
- Benchmark is past volatility alone (one or three lags), not HAR.

## My Section 4 text (all three papers, own words)
Team target ~35 words; agreed ~75–80 (tell the team: it uses some of the report's spare words).

### Current version: Draft 4 (110 words)
Christensen et al. (2023) found both Elastic Net and random forest beat HAR with extra predictors one week ahead, random forest by more, which supports testing both a linear (Elastic Net) and a tree model (LightGBM) with our extra features. Blair et al. (2001) found the VIX provided more accurate forecasts, over horizons from 1 to 20 days, than forecasts using only past returns, which supports our use of the VIX as a predictor. Christiansen et al. (2012) found credit-risk and funding-stress measures predicted volatility across asset classes at a monthly horizon; we use TLT volatility and bank-sector vs market volatility as our own proxies for funding and credit stress.

### Earlier drafts
- Draft 3 (107 words): Christensen et al. (2023) found Elastic Net beat HAR with extra predictors one week ahead, though random forest did better, which supports testing both a linear (Elastic Net) and a tree model (LightGBM) with our extra features. Blair et al. (2001) found the VIX provided more accurate forecasts, over horizons from 1 to 20 days, than forecasts using only past returns, which supports our use of the VIX as a predictor. Christiansen et al. (2012) found credit-risk and funding-stress measures predicted volatility across asset classes at a monthly horizon; we use TLT volatility and bank-sector vs market volatility as our own proxies for funding and credit stress.
- Draft 2 (84 words, all three papers): Christensen et al. (2023) found Elastic Net beat HAR with extra predictors one week ahead, though random forest did better. Blair et al. (2001) found the VIX provided more accurate forecasts, over horizons from 1 to 20 days, than forecasts using only past returns, which supports our use of the VIX as a predictor. Christiansen et al. (2012) found credit-risk and funding-stress measures predicted volatility across asset classes at a monthly horizon; we use TLT volatility and bank-sector vs market volatility as our own proxies.
- Draft 1, Christensen (34 words): Christensen et al. (2023) found that Elastic Net beat HAR with full set of predictors one week ahead (0.925, significant at 5%), though random forest did better than Elastic Net (0.879, significant at 1%).
- Draft 1, Blair (34 words): Blair et al. (2001) found that the VIX provided more accurate forecasts, over horizons from 1 to 20 days, than forecasts using only past returns, which supports our use of the VIX as a predictor.
- Draft 1, Christiansen (~95 words): Christiansen et al. (2012) analysed which variables predict volatility in four asset classes. They found that credit-risk and funding-stress measures predicted volatility across asset classes at a monthly horizon, while purely macroeconomic variables added little. They used log of the square root of realised variance from daily returns, which supports our choice to use log of SPY realised volatility from daily data. They also used the TED spread as a stress predictor. In our case, we use TLT volatility, and bank-sector vs market volatility (XLF vs SPY) as our own proxies for funding and credit stress, which the paper does not test.

## Section 5.3 Elastic Net
Zou & Hastie (2005). Regularization and variable selection via the elastic net. *JRSS Series B* 67(2), 301–320.

### What the authors did
- Proposed the elastic net: least squares with both a lasso penalty (sum of absolute coefficients) and a ridge penalty (sum of squared coefficients) (Section 2).
    - Assumes a centred response and standardised predictors (Section 2.1)
- Corrected the "naïve" version, which shrinks coefficients too much, by rescaling them by (1 + λ2) (Section 3).
- Tested it on simulated and real data (Sections 4–6).
    - Prostate cancer data: 67 training rows, 30 test rows, 8 predictors; tuning by tenfold cross-validation (CV)
    - Four simulations, 50 data sets each; tuning on a separate validation set

### What they found
- Grouping effect: strongly correlated predictors tend to enter or leave the model together (abstract; Section 2.3).
    - Theorem 1: the gap between the coefficients of predictors i and j is at most (1/λ2)·√(2(1 − ρ)), where ρ is their correlation and λ2 the ridge penalty
    - As ρ → 1, their coefficients become almost equal
    - The lasso has no grouping effect: it tends to pick one variable from a correlated group (Section 1, scenario (b))
- The elastic net is more accurate than the lasso in all four simulations (Section 5, Table 2).
    - Reduction in prediction error vs the lasso: 18%, 18%, 13%, 27%
- On the prostate data (more rows than predictors), the elastic net has the lowest test mean squared error (Section 4, Table 1).
    - Test mean squared error: elastic net 0.381; lasso 0.499; ridge 0.566; ordinary least squares (OLS) 0.586

### Links to our design (authors' facts, ours next to them)
- Standardised predictors are part of the method's setup (Section 2.1; Theorem 1).
    - Ours: StandardScaler inside the Pipeline
- Correlated predictors: the grouping effect applies when ρ is high (Theorem 1).
    - Ours: v4 reports a correlation of 0.88 between f3 and the VIX (to confirm in the T3 heatmap)
- Tuning: tenfold cross-validation on the prostate data; a validation set in the simulations.
    - Ours: TimeSeriesSplit(n_splits=5, gap=5) on training rows only

### Caveats for the Q&A
- The paper's main motivation is many more predictors than rows (e.g. gene data); we have 2,452 training rows and 10 features.
    - Scenario (c) in Section 1 covers more rows than predictors, with highly correlated predictors
- TODO (outside the material, not verified): check in the scikit-learn docs whether ElasticNet applies the (1 + λ2) rescaling or fits the naïve version.
