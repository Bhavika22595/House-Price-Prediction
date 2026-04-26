House Price Prediction - Hyderabad
A college team project where we predicted residential property
prices in Hyderabad using data we scraped ourselves from
MagicBricks and trained a model on Azure ML Studio.

Why we built this?
Buying a house in Hyderabad is confusing — two similar flats in
the same area can be listed at very different prices and there's
no easy way to know what's actually fair. We wanted to build
something that takes basic property details and gives a
data-backed price estimate.

Tech used:
Python, PyCharm
Selenium, webdriver-manager
Pandas, NumPy
Azure Machine Learning Studio (Designer)
Postman

Step 1 — Scraping the data
We scraped Residential Houses and Villas listed for sale in
Hyderabad from MagicBricks. The URL we scraped covered 1BHK
to 5BHK+ across both property types:
magicbricks.com/property-for-sale/residential-real-estate?
bedroom=>5,1,2,3,4&proptype=Residential-House,Villa&cityName=Hyderabad

MagicBricks loads listings dynamically with JavaScript so
BeautifulSoup doesn't work here — it only sees the raw HTML
before JS runs. Selenium controls an actual Chrome browser so
it waits for everything to load properly.
How the scraper works:
The results page shows listing cards. Clicking a card opens the
full property detail page in a new tab — the card itself doesn't
have all the info we needed. So the scraper:

Finds all listing cards on the page
Reads the title, extracts locality name by splitting on "in"
(titles follow the format "3 BHK Villa in Bachupally")
Checks if the card shows CARPET AREA — skips it if not
(some cards show Super Area which is a different measurement,
mixing the two would make the area column inconsistent)
Reads area from the card, converts sqyrd to sqft if needed
(1 sqyrd = 9 sqft)
Clicks the listing → new tab opens
Switches to new tab, scrapes all the details
Closes tab, switches back to results page
After going through all cards, scrolls down to load more
Stops when scrolling doesn't load any new listings

Handling popups:
MagicBricks shows an NPS survey popup randomly that sits on top
of listings and blocks clicks. We handled it with a retry loop —
try clicking, if ElementClickInterceptedException happens close
the popup and retry, up to 3 attempts before skipping.
Avoiding duplicates:
After scrolling, previously loaded cards stay on the page.
Without handling this we'd scrape the same listing multiple
times. We kept a processed_addresses set and skipped any title
we'd already seen.
Columns collected:
Price | Bedrooms | Bathrooms | Balconies | Carpet Area (sqft) |
Place | Furnished Status | Near Main Road | Transaction Type |
Car Parking | Facing
Place is the locality extracted from the listing title.
Near Main Road is Yes/No based on whether "Main Road" appeared
in the expanded property details section.
Transaction Type is either Resale or New Property.
Output saves to: ~/Downloads/magicbricks_properties.csv

Step 2 — Cleaning the data
Done in Python before uploading to Azure.
Price standardization
MagicBricks shows prices in different formats — "₹ 45.5 Lac",
"₹ 1.2 Cr", "Call for Price". We converted everything to a
plain rupee number and dropped "Call for Price" rows:
pythondef convert_to_number(price_txt):
    if 'Call for Price' in price_txt:
        return None
    elif 'Cr' in price_txt:
        return float(price_txt.replace('₹','').replace('Cr','').strip()) * 10_000_000
    elif 'Lac' in price_txt:
        return float(price_txt.replace('₹','').replace('Lac','').strip()) * 100_000
    else:
        return float(price_txt.replace('₹','').strip())
Other cleaning steps:

Removed duplicate rows (same listing scraped across multiple
scroll sessions)
Removed outliers using IQR method on Price and Carpet Area —
listings priced at ₹5 Cr for a 400sqft flat are clearly wrong
Missing values — around 15% of rows had at least one null.
Kept these for Azure to handle rather than dropping entire rows


Step 3 — Azure ML Pipeline
We used Azure ML Studio's Designer to build the pipeline
visually (drag and drop modules, connect them with arrows).
Azure setup:

Created an Azure for Students account ($100 free credits)
Created a Resource Group and ML Workspace, region: South India
Registered our cleaned CSV as a Tabular Dataset
Attached a Standard_DS2_v2 compute instance to run the pipeline

Pipeline steps in order:
Dataset
    → Select Columns
        removed columns with no predictive value
    → Clean Missing Data
        median imputation for numerical columns
        mode imputation for categorical columns
    → Edit Metadata
        marked Price as the label column
        tagged categorical columns explicitly
    → Apply SQL Transformation
        created 3 new features (see below)
    → Normalize Data
        min-max scaling on numerical columns
    → Convert to Indicator Values
        one-hot encoding for categorical columns
        (Furnished Status, Transaction Type, Facing, etc.)
    → Split Data
        80% train (16,000 rows), 20% test (4,000 rows)
        random seed 42 for reproducibility
    → Train Model ← Boosted Decision Tree Regression
    → Score Model
    → Evaluate Model
Feature engineering inside the pipeline:
Used the Apply SQL Transformation module to create 3 new columns:
sqlSELECT *,
    Price / [Carpet Area (sqft)]        AS price_per_sqft,
    Bedrooms + Bathrooms                AS total_rooms,
    CAST(Balconies AS FLOAT) / Bedrooms AS balcony_ratio
FROM t1;
price_per_sqft normalizes price across different sized properties.
total_rooms captures overall space in one number.
balcony_ratio gives a sense of how open the property feels
relative to its size.

Step 4 — Models we tried
We tried 4 algorithms and compared them:
ModelR²Linear Regression0.61Decision Tree Regression0.72Neural Network Regression0.69Boosted Decision Tree Regression0.85
Linear Regression underfit badly — property prices aren't
linear. A 4BHK isn't exactly twice the price of a 2BHK.
Boosted Decision Tree handled these non-linear patterns well
because it builds trees sequentially where each tree fixes
the errors of the previous one.
Hyperparameters we used:
ParameterValueNumber of trees100Learning rate0.1Number of leaves20Min samples per leaf10

Step 5 — Evaluation
After the pipeline ran, right-clicked the Evaluate Model
module → Visualize to see the metrics:
MetricValueR²0.85MAE~₹2-3 LakhsRMSE~₹4-5 Lakhs
RMSE is higher than MAE because it squares errors before
averaging — luxury properties that are hard to predict pull
it up more.

Step 6 — Deployment
After training we created a Real-time Inference Pipeline in
Azure (removes training-specific modules, adds Web Service
Input/Output). Deployed it to Azure Container Instance (ACI)
which gave us a REST endpoint with a scoring URL and API key.
Tested it in Postman — POST request with property details,
get predicted price back.
Request:
json{
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
Response:
json{ "result": [7500000] }
Postman was just for testing — it simulates what a real
frontend would send to the API. The actual model is hosted
on Azure.

Project structure
├── scraping/
│   └── scraper.py
├── preprocessing/
│   └── clean_data.py
├── data/
│   ├── raw_data.csv
│   └── clean_data.csv
├── azure/
│   └── pipeline_screenshot.png
└── README.md

Running the scraper
bashpip install selenium webdriver-manager pandas numpy
python scraping/scraper.py
Chrome must be installed. Output saves to
~/Downloads/magicbricks_properties.csv
