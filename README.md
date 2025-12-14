# Meridian Brand Exploration (Non-Coder Friendly)

This folder holds the Hotel Brands survey export and guided notebooks. You can follow the steps and open the ready-made tables in Excel without writing code.

## Quick Start (non-coders)
1. Activate the `venv` (if not already): `source venv/bin/activate`.
2. Open the main notebook: `jupyter notebook meridien_clarity.ipynb` (or open in VS Code).
3. In the notebook, click “Run All”. It will:
   - Load the survey file,
   - Generate charts you can read in-place,
   - Save clean tables into the `exports/` folder for Excel.
4. Open any CSV in `exports/` with Excel to build your own charts (e.g., `exports/meridien_overlap_means.csv`).

## Key Notebooks
- `meridien_clarity.ipynb` – Primary, plain-language walkthrough. Each section explains what it does, shows the result, and writes a CSV you can open in Excel.
- `meridien.ipynb` – Earlier deep dive (kept for reference).
- `first exploration.ipynb` – Quick helper-driven pass over the full dataset.

## Helper Utilities
- `count_nulls(df)` – Table of missing values.
- Plot helpers (`ajr_plot_histograms`, `ajr_plot_correlations`, `ajr_correlation_heatmap`) – available if you want to extend visuals beyond the clarity notebook.
- Modeling helpers (`ajr_test_models`, `ajr_find_best_combination`) – unused in the clarity notebook but ready for future work.

## Meridien Feature Glossary
- `Meridiennights`: Total nights the respondent spent at any Le Meridien during the recall window.
- `safeMeridien`: Agreement rating for “I feel safe at Le Meridien.”
- `sucMeridien`: Agreement rating for “Le Meridien helps me be successful/get things done.”
- `specMeridien`: Perception that Le Meridien feels special or distinctive.
- `cfreeMerid`: “Carefree/relaxed at Le Meridien” agreement rating.
- `wellMerid`: “Le Meridien makes me feel welcomed/well” sentiment.
- `clMerid`: Cleanliness/clarity perception for the brand.
- `wmMerid`: Warm & modern impression (“warm modern” shorthand).
- `pamMerid`: Feeling of being pampered or indulged at Le Meridien.
- `Meridienpen`: Penalty score—how strongly respondents said they would avoid the brand (higher = worse).
- `Meridienpf`: Preference score—how often Le Meridien is the first-choice brand (higher = better).
- `SCRmerid`: Share of consideration/recommendation (0–1 advocacy score for the brand).

## Using the exports
After running `meridien_clarity.ipynb`, check the `exports/` folder for ready-to-open CSVs:
- `null_overview.csv` – top missing columns.
- `trip_summary.csv` – business vs. leisure vs. total nights.
- `meridien_nights_summary.csv` – Meridien stay counts.
- `meridien_overlap_means.csv` – average stays with other brands among Meridien guests.
- `meridien_attitude_summary.csv` and `meridien_correlations.csv` – opinion stats and their link to `SCRmerid`.
- `scr_summary.csv` and `scr_means.csv` – advocacy scores by brand.
- `meridien_intent_summary.csv` – penalty and preference for Meridien.
- `segment_summary.csv` – business vs. leisure segment view.

## Headline Findings (from the clarity notebook)
- Reach: 361 of 385 respondents logged ≥1 Meridien stay, but only 19 answered `SCRmerid`, so low engagement (not dislike) drives the weak average score (~0.047).
- Drivers: Warmth/safety statements (`wellMerid`, `safeMeridien`, `clMerid`) show the strongest link to higher `SCRmerid`; product/price items barely register.
- Brand adjacency: Meridien guests also average ~2 Hilton and ~1.8 Marriott stays annually—prime loyalty overlap.
- Intent: Penalty mean is ~0.05 (almost nobody refuses the brand) but preference mean is ~4/7, indicating indifference rather than rejection.
- Segments: Leisure-leaning travelers report the highest Meridien SCR (~0.056) despite low stay counts; leisure-forward offers are the fastest lever.

### About sample sizes
- Each section in `meridien_clarity.ipynb` states the exact base it uses. Some views use everyone (travel mix), some use Meridien stayers (overlap), and the attitude/SCR correlations use only the small set of people who answered every Meridien question. SCR comparisons use whoever answered each brand’s SCR, so counts differ by brand.

Need to tweak visuals or tables? Open `meridien_clarity.ipynb`, edit text cells, and rerun—no coding required.
