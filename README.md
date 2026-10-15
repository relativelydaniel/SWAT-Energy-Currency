# DSR ENERGY x CURRENCY CASE STUDY

DS / ML project. Emphasis on API maintenance and ETL, Exploratory Data Analysis and and Machine Learning Prediction.

QUESTION: “To what extent could asymmetrical Solar and Electrical energy weighting help model West African currency stability by a 2030 based scenario?”

Dataset: API driven UNComtrade data, DBNomics

Tech Stack: Python, SQL, NashPy, SKlearn, Stats Model (data cleaning, intro / intermediate statistics + game theory, ML models)

The Digital Silk Road adding cultural context, I wanted to explore the logistical implications between China, Nigeria and Ghana properly after my previous question was too niche and was microeconomic. Also added a game theoretical approach to rein in the scenario 

My ETL skills improved as I used basic automation and the psycog library to push API data to a PGSQL database. The ETL script also included ML based imputation to accurately fill missing values, and a fallback clause. 

I’m currently in the process of parametrising ML models (Linear Regression, PCA based weighting, Payoff Matrix, etc.)

FUTURE: Host the stream of visuals on a Cloudflare powered website and Streamlit 

Future culmination with a simpler, preceding EDA and Visualisation project into a Clouflare hosted website via a JSON / API bridge
