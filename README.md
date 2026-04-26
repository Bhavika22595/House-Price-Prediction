# House Price Prediction - Hyderabad

A college team project where we predicted residential property prices in Hyderabad using data we scraped ourselves from MagicBricks and trained a model on Azure ML Studio.

---

## Why we built this

Buying a house in Hyderabad is confusing — two similar flats in the same area can be listed at very different prices and there's no easy way to know what's actually fair. We wanted to build something that takes basic property details and gives a data-backed price estimate.

---

## Tech used

- Python, PyCharm
- Selenium, webdriver-manager
- Pandas, NumPy
- Azure Machine Learning Studio (Designer)
- Postman

---

## Step 1 — Scraping the data

We scraped Residential Houses and Villas listed for sale in Hyderabad from MagicBricks. The URL we scraped covered 1BHK to 5BHK+ across both property types:

magicbricks.com/property-for-sale/residential-real-estate?bedroom=>5,1,2,3,4&proptype=Residential-House,Villa&cityName=Hyderabad

MagicBricks loads listings dynamically with JavaScript so BeautifulSoup doesn't work here — it only sees the raw HTML before JS runs. Selenium controls an actual Chrome browser so it waits for everything to load properly.

**How the scraper works:**

The results page shows listing cards. Clicking a card opens the full property detail page in a new tab — the card itself doesn't have all the info we needed. So the scraper:

1. Finds all listing cards on the page
2. Reads the title, extracts locality name by splitting on "in" (titles follow the format "3 BHK Villa in Bachupally")
3. Checks if the card shows CARPET AREA — skips it if not (some cards show Super Area which is a different measurement, mixing the two would make the area column inconsistent)
4. Reads area from the card, converts sqyrd to sqft if needed (1 sqyrd = 9 sqft)
5. Clicks the listing → new tab opens
6. Switches to new tab, scrapes all the details
7. Closes tab, switches back to results page
8. After going through all cards, scrolls down to load more
9. Stops when scrolling doesn't load any new listings

**Handling popups:**

MagicBricks shows an NPS survey popup randomly that sits on top of listings and blocks clicks. We handled it with a retry loop — try clicking, if `ElementClickInterceptedException` happens close the popup and retry, up to 3 attempts before skipping.

**Avoiding duplicates:**

After scrolling, previously loaded cards stay on the page. Without handling this we'd scrape the same listing multiple times. We kept a `processed_addresses` set and skipped any title we'd already seen.

**Columns collected:**

| Column | Description |
|---|---|
| Price | Listing price converted to rupees |
| Bedrooms | Number of bedrooms |
| Bathrooms | Number of bathrooms |
| Balconies | Number of balconies |
| Carpet Area (sqft) | Usable floor area in sqft |
| Place | Locality extracted from listing title |
| Furnished Status | Furnished / Semi-Furnished / Unfurnished |
| Main Road | Yes/No based on property details section |
| Transaction Type | Resale or New Property |
| Car Parking | Number of covered parking spots |
| Facing | Direction the property faces |

Output saves to: `~/Downloads/magicbricks_properties.csv`

---

## Step 2 — Cleaning the data

Done in Python before uploading to Azure.

**Price standardization**

MagicBricks shows prices in different formats — `₹ 45.5 Lac`, `₹ 1.2 Cr`, `Call for Price`. We converted everything to a plain rupee number and dropped "Call for Price" rows:

```python
def convert_to_number(price_txt):
    if 'Call for Price' in price_txt:
        return None
    elif 'Cr' in price_txt:
        return float(price_txt.replace('₹','').replace('Cr','').strip()) * 10_000_000
    elif 'Lac' in price_txt:
        return float(price_txt.replace('₹','').replace('Lac','').strip()) * 100_000
    else:
        return float(price_txt.replace('₹','').strip())
```

**Other cleaning steps:**

- Removed duplicate rows (same listing scraped across multiple scroll sessions)
- Removed outliers using IQR method on Price and Carpet Area — listings priced at ₹5 Cr for a 400sqft flat are clearly wrong
- Missing values — around 15% of rows had at least one null. Kept these for Azure to handle rather than dropping entire rows

---

## Step 3 — Azure ML Pipeline

We used Azure ML Studio's Designer to build the pipeline visually (drag and drop modules, connect them with arrows).

**Azure setup:**
- Created an Azure for Students account ($100 free credits)
- Created a Resource Group and ML Workspace, region: South India
- Registered our cleaned CSV as a Tabular Dataset
- Attached a Standard_DS2_v2 compute instance to run the pipeline

**Pipeline steps in order:**
Dataset
→ Select Columns          (removed columns with no predictive value)
→ Clean Missing Data      (median for numbers, mode for text)
→ Edit Metadata           (marked Price as label column)
→ Apply SQL Transformation (created 3 new features)
→ Normalize Data          (min-max scaling)
→ Convert to Indicators   (one-hot encoding for categorical columns)
→ Split Data              (80% train / 20% test, seed 42)
→ Train Model             ← Boosted Decision Tree Regression
→ Score Model
→ Evaluate Model

**Feature engineering inside the pipeline:**

Used the Apply SQL Transformation module to create 3 new columns:

```sql
SELECT *,
    Price / [Carpet Area (sqft)]         AS price_per_sqft,
    Bedrooms + Bathrooms                 AS total_rooms,
    CAST(Balconies AS FLOAT) / Bedrooms  AS balcony_ratio
FROM t1;
```

- `price_per_sqft` — normalizes price across different sized properties
- `total_rooms` — captures overall space in one number
- `balcony_ratio` — gives a sense of how open the property feels relative to its size

---

## Step 4 — Models we tried

| Model | R² |
|---|---|
| Linear Regression | 0.61 |
| Decision Tree Regression | 0.72 |
| Neural Network Regression | 0.69 |
| **Boosted Decision Tree Regression** | **0.85** |

Linear Regression underfit badly — property prices aren't linear. A 4BHK isn't exactly twice the price of a 2BHK. Boosted Decision Tree handled these non-linear patterns well because it builds trees sequentially where each tree fixes the errors of the previous one.

**Hyperparameters used:**

| Parameter | Value |
|---|---|
| Number of trees | 100 |
| Learning rate | 0.1 |
| Number of leaves | 20 |
| Min samples per leaf | 10 |

---

## Step 5 — Evaluation

After the pipeline ran, right-clicked the Evaluate Model module → Visualize to see the metrics:

| Metric | Value |
|---|---|
| R² | 0.85 |
| MAE | ~₹2-3 Lakhs |
| RMSE | ~₹4-5 Lakhs |

RMSE is higher than MAE because it squares errors before averaging — luxury properties that are hard to predict pull it up more.

---

## Step 6 — Deployment

After training we created a Real-time Inference Pipeline in Azure (removes training-specific modules, adds Web Service Input/Output). Deployed it to Azure Container Instance (ACI) which gave us a REST endpoint with a scoring URL and API key.

Tested it in Postman — POST request with property details, get predicted price back.

**Request:**

```json
{
  "data": [{
    "Carpet Area (sqft)": 1200,
    "Bedrooms": 3,
    "Bathrooms": 2,
    "Balconies": 1,
    "Furnished_status": "Semi-Furnished",
    "Transaction Type": "Resale",
    "Main Road": "Yes",
    "Car Parking": 1,
    "Facing": "East",
    "Place": "Bachupally"
  }]
}
```

**Response:**

```json
{ "result": [7500000] }
```

Postman was just for testing — it simulates what a real frontend would send to the API. The actual model is hosted on Azure.

---

---

## Running the scraper

```bash
pip install selenium webdriver-manager pandas numpy
python scraping/scraper.py
```

Chrome must be installed. Output saves to `~/Downloads/magicbricks_properties.csv`

---

## What we'd do differently

- Replace `time.sleep()` with Selenium's `WebDriverWait` — the fixed 5 second waits slow everything down, explicit waits would only wait as long as actually needed
- Make the scraper resumable — if it crashes the `processed_addresses` set resets on restart so it re-scrapes from the top. Should load already-scraped titles from the CSV at startup
- Cross-validation instead of a single train-test split
- Add distance to metro/IT parks as a feature — location quality matters a lot for Hyderabad prices
- Build a small web frontend instead of testing via Postman

---

Built using Azure for Students free credits. Team of 3.
