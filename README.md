# QUB-dissertation
AIRBNB MARKET INVESTIGATIONS: A COMPARATIVE STUDY OF PRICING, HOST BEHAVIOUR AND BUSINESS MODELS IN MELBOURNE AND SYDNEY

Acknowledgements
First and foremost, I would like to extend my heartfelt appreciation to my supervisor, Dr. Olivia Peng, for her tremendous supervision and infinite patience throughout the dissertation process. Dr. Peng's expertise and detailed suggestions were crucial during the final phases of this work, and I consider myself very fortunate to have learned from a dedicated mentor. There are so many incredible aspects of Queen's Business School, and I continue to appreciate being part of a fast-paced, stimulating academic experience. I am especially thankful to Dr. Byron Graham for the guidance and support for the MSc. Business Analytics course. Finally, I owe so much to my family and friends, and I would like to separately recognize Hom Nan Thawe Htun, who has supported me throughout the process. Their endless belief and encouragement provided me the strength to carry my dissertation through to fruition. This work is equally theirs as it is mine.

Abstract
The rapid advancement of Airbnb has disrupted the rental market in Australia, primarily in Melbourne and Sydney, and raised questions about housing affordability and market dynamics. This research explored the predictors of pricing and availability in these two cities as a function of the changing behaviors of accessible factors or traits using machine learning, geospatial analytics, and Explainable AI (XAI). We implemented the CRISP-DM functionality by assessing public-domain datasets derived from InsideAirbnb applying various modelling techniques (XGBoost, Random Forest, Gradient Boosting, KNN, and OLS) to predict pricing and occupancy. XGBoost was identified as the top model, with test R² values of 0.4998 (Melbourne) and 0.5118 (Sydney) along with lower error metrics in Melbourne (RMSE: 86.0365), indicating further predictability within the market. Through Principal Component Analysis and subsequent K-means clustering analysis, we uncovered various market segments, which indicated four clusters in Sydney (e.g., professional hosts, budget listings), while Melbourne had five clusters, including extra-large property types in particular. Upon further explorative analysis using the Shapley Additive exPlanations (SHAP) techniques, it was indicated that the primary factors, which predicted price included the number of bedrooms, number of accommodates, and the impact of geospatial analysis including longitude. It was concluded that while housing markets were relatively predictable for both cities due to the overwhelmed number of hosts, Melbourne was argued to be more predictable due to the compactness of the urban area, contrasted with the more variability in the Sydney market which was based on the market characteristics of fragmented demand.

1.0 Introduction
In recent years, the peer-to-peer (P2P) economy has a dramatic transformation, which facilitates
customers and service providers to engage in direct online transaction. Airbnb exemplifies this
change, having redefined the travel and tourism sector by offering a digital platform that
connects individuals searching for accommodations with those looking to rent out their
properties. Deloitte (2017) conveyed Airbnb span over 65,000 cities and towns while Del
Giudice et al. (2024) noted that the platform hosts over 5 million listing in 191 countries.
Therefore, it can be seen that the impact of Airbnb has significantly emerged within a few years.
As of 2024, the company recorded $11 billion revenue that is a 12% year-over-year growth
over the past year. Over and above that, it enabled approximately 491 million reservations
generating a total gross booking value of $82 billion (Airbnb, 2025).
As stated by Milunovich and Nasrabadi (2024), Australia alone has more than 250,000 short-
term rental listings, of which about a quarter are in Sydney. In examining Hill et al. (2023),
Airbnb faces short-term rental markets through the imposition of a day-limit regulation on their
listings. It has reshaped the markets in big cities such as Melbourne and Sydney, fuelling
housing affordability and neighbourhood concerns. Property values and rents keep increasing,
and vacancy rates continue to be low. As a result, a number of Australian cities are now among
the most unaffordable housing markets in the world. (Urban Reform Institute, 2023; Cox, 2024).
Airbnb provides a diverse selection of accommodations, from entire homes to private rooms,
accommodating to variety of traveller demands and budgets. The frequently found Airbnb
rentals can be seen in high-demand regions like central Melbourne, Docklands, Southbank,
Fitzroy, St Kilda, Bondi, Manly and Darlinghurst. These areas arrange easy access to public
transformations, employment hubs, and tourist attractions, making Airbnb an appealing choice
for travellers seeking accessibility to metropolitan amenities or iconic places (Airbnb, 2024).
It is also more cost-effective in comparison to hotels, particularly for those needing larger
accommodation or longer stays (Lin, 2020). While hotel rates can be exorbitant owing to
tourism demand, Airbnb is yet another platform with a different reasonable price. From the
residents' point of view, hosting on Airbnb provides monetary rewards, indirectly promoting
the platform's popularity. In addition, Gomes (2022) stated that the majority of the guests are
opting to stay the location which provides home-like experience. Most of the hosts can utilize
the revenue to cover exorbitant living expenses or mortgage repayments in such costly cities.
According to the Urbis report (2023), more than 90% of Airbnb hosts in Australia reside in the
same state as their property listing and utilize the revenue for economic, social, or wellbeing
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 6
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
purposes, including meeting new individuals or complementing discretionary expenditure. The
economic incentive maintains a healthy supply of listings, providing travellers with numerous
options.
With the rapid growth and revolutionary impact on hospitality, housing, and urban economies,
this study focuses on Airbnb, with particular attention to the cities of Melbourne and Sydney.
The cities were selected due to their significant Airbnb activity, diverse market conditions
influenced by urban density and tourism, and the pressing housing affordability challenges they
face (Alizadeh et al., 2018). By comparing the two cities, the study examines how the growth
of Airbnb impacts socio-economic equity in Australia's two biggest urban areas. Although both
cities are major tourist centers facing rising housing pressures, Sydney has almost double the
number of Airbnb listings, which are highly concentrated in wealthy areas like coastal suburbs,
while Melbourne has a more dispersed pattern (Alizadeh et al., 2018). The research examines
these differences in listing concentrations, rental market pressures, and host demographics—
like education, as a key indicator shaping participation—to explain how the urban context
mediates Airbnb's effects. The results not only assist in the identification of clusters and market
segmentation via data analysis, but also indicate areas in which regulatory efforts, like
restrictions on short-term rentals, are most needed to mitigate housing displacement and
inequality.
Thus, because of the fast growth of Airbnb has revolutionized urban housing markets and
tourism economies globally, but gaps persist in knowledge of how listing attributes, host
practices, and geospatial variables jointly influence pricing and availability dynamics—
especially in comparative analyses between cities. This research fills these gaps by exploring
Melbourne and Sydney, Australia's most extensive and most Airbnb-saturated metropolitan
cities, where housing affordability crises and regulatory controversies amplify demands for
data-informed insight. Through the application of machine learning, geospatial analysis, and
Explainable AI (XAI), the study seeks to reveal the complex drivers of Airbnb markets while
facilitating interpretability for stakeholders.
One of its significant contributions is in its comparative design, which systematically tests how
urban context (Sydney's beachside clustering vs. Melbourne's scattered pattern) mediates the
effect of attributes such as property characteristics, host reputation, and distance to amenities.
In contrast to earlier single-city studies, this study applies unsupervised learning (PCA, K-
means clustering) to discover unique market segments and host strategies in each city,
unearthing localized traveler priorities and competitive landscapes. Additionally, it pushes
methodological bars by comparing predictive models (e.g., XGBoost, Random Forest) with
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 7
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
cross-validation and interpreting findings using XAI methods such as SHAP and LIME. These
not only measure feature importance but also unveil nonlinear interactions, e.g., how
"Superhost" status in high-demand areas disproportionately increases prices, providing
actionable insights for hosts to optimize pricing.
For policymakers, the research offers evidence to shape regulations targeting housing shortages.
By mapping out concentrations of "commercial" hosts (those with multiple listings), for
example, it identifies where short-term rentals are likely to put additional pressure on
affordability. In connecting technical analysis to applied use cases, this study speaks to both
academic scholarship and actual decision-making in the sharing economy.
1.1 Report Structure
The report is structured to analyse Airbnb market dynamics in Melbourne and Sydney using
machine learning and geospatial analytics by stating Section 2 reviews the relevant Literature
Review to summarize existing research on Airbnb related to the market prices and machine
learning models. Section 3 describes the dataset, variable selection, and data preprocessing
procedures following CRISP-DM phases including data understanding, data preparation, data
cleaning, feature selection, modelling, evaluation and interpretability machine learning. Section
4 presents the modelling methodology, including the selected algorithms and the SHAP-based
interpretation framework. Section 5 discusses empirical results and identifies key findings
related to model performance, PCA, clustering, and feature importance. Section 6 interprets the
results in the context of research questions and existing literature. Finally, Section 7 concludes
the paper by summarizing the key findings and provide recommendations for Airbnb hosts and
policymakers with the highlights of limitation.
2.0 Literature Review
2.1 Key Findings on Airbnb Market Characteristics
Due to the growing demand in the Airbnb market, this study synthesizes current academic
literature of Airbnb market research.
Liao & Nawaz's (2024) investigation of Hong Kong's Airbnb market uncovered important
findings regarding location choice in shared accommodation. Their research illustrated that
built environment attributes - such as land use configurations, urban function, and transport
accessibility - play a pronounced role in where Airbnb properties are clustered. The study also
indicated how socioeconomic characteristics and topographic limitations affect the spatial
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 8
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
pattern of short-term rentals, with properties focusing in locations that compromise between
tourist demand and residential convenience.
Panahandeh et al. (2025) found that the incorporation of geographic features, such as proximity
to city centers, universities, and nearby available services, significantly enhanced the accuracy
of Airbnb price predictions in Dublin. The SHAP analysis in the research confirmed the
significant effect of geographic, property, and host-related factors on rental prices, highlighting
their vital role in influencing market trends.
Gao (2025) discovered that when forecasting London Airbnb prices, accommodation capacity
variables, including the number of guests, bathrooms, and bedrooms, were the most dominant
predictors, which is consistent with Alharbi (2023) and Gibbs et al. (2017). Host activity, room
type, and recency of reviews also played significant roles in pricing. Although external tourism
and environmental factors were highly ranked in feature importance, their overall effect on
model performance was minimal, indicating possible overlap with essential location-based
features.
Meanwhile, Pittala et al. (2024) identified key pricing determinants for Airbnb across European
cities, including proximity to tourist attractions, property type, amenities, and host-related
factors. The analysis revealed that these elements significantly influence pricing patterns,
providing insights into the market dynamics driving rental values.
Moreno-Izquierdo et al. (2018) showed that in the Valencian Community, comprising Valencia,
Alicante, and Castellón, neural networks significantly outperformed conventional hedonic
pricing models in their ability to forecast Airbnb prices. This superiority stemmed from the
models' ability to detect complex, nonlinear relationships often overlooked by standard
methods, thereby producing more accurate price predictions. The study nonetheless recognized
some drawbacks regarding interpretability and stability of the models, highlighting areas where
market insights could be obfuscated by the non-transparent nature of neural networks.
Gibbs et al. (2018) determined several key determinants of Airbnb listing prices in five major
Canadian cities—Toronto, Montreal, Ottawa, Calgary, and Vancouver—by employing a
hedonic pricing model. Key factors included property type, location, number of available
amenities, and host professionalism and experience. Listings managed by more experienced
hosts, as well as those offering entire homes, tended to charge higher prices. Privacy was also
priced at a premium; however, it was interesting that higher review scores were unexpectedly
associated with lower prices, potentially reflecting a greater demand for competitively priced,
high-value options. Moreover, proximity to downtown consistently impacted pricing structures,
with prices decreasing as the distance from such locations grew farther, while Superhost
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 9
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
designation positively impacted prices in some markets, though independent properties often
had lower pricing than multi-unit homes. The study also revealed inconsistent pricing practices
across the cities under consideration, reflecting a need for greater educational services and
pricing tools for hosts. Interestingly, this result is contrary to Nguyen (2020), who found that
hedonic models exhibited predictive capability within Vietnam's housing market, thus
highlighting potential contextual differences.
Last but not least, Milunovich and Nasrabadi (2025) made clear in their Sydney-focused study
that ensemble approaches provided extremely accurate Airbnb price predictions while
determining key determinants like guest capacity, luxurious amenities, property type, and
proximity to points of interest using SHAP analysis. The study revealed patterns related to
policy, establishing that about 37% of listings were managed by professional hosts, thus
reflecting the need for differentiated regulatory strategies. In addition, it supported previous
research by Zhu et al. (2020), Ghosh et al. (2023), and Dhillon et al. (2021), which also
highlighted the greater effectiveness of ensemble techniques in Airbnb price prediction in some
of the world's urban hubs.
2.2 Methodological Approaches in Airbnb Research
Liao & Nawaz (2024) advanced methodological approaches through their application of
multiscale geographically weighted regression (MGWR). This technique proved particularly
effective by accounting for spatial non-stationarity in the relationships between variables and
allowing different bandwidths for different predictors to capture varying spatial scales and
outperforming both traditional OLS regression (by 32% in R²) and standard GWR (by 18% in
R²).
The study of Panahandeh et al. (2025) utilized XGBoost modeling with 10-fold cross-validation
to evaluate price predictions, observing an approximate 2% improvement in the R² score after
adding geographic features. Shapley Additive exPlanations (SHAP) analysis was employed to
assess and confirm the impact of various factors on pricing.
Gao (2025) study focused on predicting London Airbnb prices using Random Forest, which
demonstrated superior predictive performance despite signs of overfitting. XGBoost or other
models were implied but not specified as primary; the analysis highlighted Random Forest's
strengths in handling the dataset's features.
The Pittala et al. (2024) study employed Multivariate Linear Regression, Polynomial
Regression, Lasso Regression, and Random Forest to analyze Airbnb pricing across European
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 10
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
cities. Among these, Random Forest demonstrated superior predictive accuracy, achieving a
significantly higher R-squared value of 0.76 and a lower Mean Squared Error (MSE) of 140.72
compared to linear and polynomial regression methods, confirming its effectiveness in
modelling the dataset.
The research Moreno-Izquierdo et al. (2018) utilized neural networks to optimize Airbnb
pricing strategies, leveraging data from Valencia, Alicante, and Castellón. The study compared
these neural network models against traditional hedonic pricing models, demonstrating their
enhanced predictive performance. To address limitations in interpretability and stability, the
authors suggested integrating boosting techniques for future research to refine the models'
effectiveness.
Gibbs et al. (2018) applied a hedonic pricing model to analyse Airbnb listings across five major
Canadian cities: Toronto, Montreal, Ottawa, Calgary, and Vancouver. The regression-based
approach focused on examining factors influencing listing prices, achieving reasonable
explanatory power. However, the study did not benchmark its performance against alternative
predictive methods, leaving uncertainty about whether the hedonic model was the optimal
approach.
The Milunovich and Nasrabadi (2025) research combined machine learning and explainable AI
techniques to predict Airbnb prices in Sydney, with ensemble methods consistently
outperforming traditional and individual machine learning models across multiple performance
metrics, including RMSE, MAE, and MAPE. Feature engineering enhanced predictive
accuracy, particularly in training data, though improvements were more modest in test datasets,
indicating a potential risk of overfitting. SHAP analysis was employed to interpret the model
and identify significant price determinants.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 11
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Table 1: Comparative table of Airbnb Price Prediction Studies
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 12
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
2.3 Research Gaps
While prior studies have explored Airbnb pricing using hedonic models, regression techniques,
and advanced machine learning algorithms, many lacked comparative benchmarking or
transparency in model interpretation. Few studies combined machine learning with Explainable
AI (XAI) to enhance both predictive performance and interpretability. Moreover, limited
research has focused on integrating spatial, host, and property-level features in Australian
markets like Sydney and Melbourne. Building upon previous research, this study adopts
a combination of machine learning algorithms (Linear Regression, Lasso Regression, Random
Forest, XGBoost, Support Vector Regression (SVR), MGWR, gradient boosting) and data
mining to account for the value of property attributes and employs XAI methods to enhance
model transparency and interpretability, thereby supporting more informed decision-making.
3.0 Data and Methodology
This report deployed the Cross-Industry Standard Process for Data Mining (CRISP-DM)
framework, which extracts meaningful insights from data by offering a structured and iterative
process (Gao, 2025). In order to reach this outcome, there are six steps will be considered upon
CRISP-DM method as business understanding, data understanding, data preparation, modelling,
evaluation and deployment (Chen, 2024).
Figure 1: CRISP-DM Process
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 13
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
3.1 Business Understanding
The preliminary phase of this research analyzing listing characteristics, host behavior, and
location-specific factors in Sydney and Melbourne. Utilizing regression models (XGBoost,
Gradient Boosting, Random Forest, KNN, Multiple Linear Regression) and unsupervised
learning, the study identifies key pricing drivers, host behavioral groups, and market segments.
Regression analysis reveals significant pricing factors, market trends, and enables price
forecasting, supporting strategic decision-making for competitive pricing and revenue
optimization (Quattrone et al., 2018). Sydney listings concentrate around the central business
district and coastal suburbs, while Melbourne’s extend across inner and outer suburbs (Figure
2).
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 14
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 2: The Distribution of Airbnb in Sydney and Melbourne
3.2 Data Understanding
This section is a fundamental part in data mining process for analysis and modelling. Based on
this, it allows to explore data structure, access its data quality and evaluate its relevance to the
research objectives (Gao, 2025). The dataset is deployed using public-domain data from Inside
Airbnb accessing Google colab's pandas library. The starting host year is from 2013 which is 5
years of Airbnb establishment. The below Table 2 is presented as feature descriptions.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 15
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Table 2: Features Description
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 16
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 17
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
3.3 Data preparation
The data preparation step includes initial data exploration, data cleaning, data conversion,
outliers and exploratory data analysis (EDA). This phase ensures the dataset to be suitable for
data accuracy, analysis and modelling. In this process, the dataset was initially examined its
structure, data types and completeness. Throughout the cleaning procedures, missing values
and discrepancies were addressed. Categorical and textual variables were transformed into
numerical formats. To avoid distortion in the modelling stage, the outliers were identified and
solved. Lastly, EDA approaches with visual aids such as correlation, box plots and histograms
were applied to define patterns and the key variables relationships. The specifics steps are
provided in the following subsections.
3.3.1 Initial Data Exploration
The initial exploration involved examining the structure, types, skewed distributions and
completeness of the dataset. The variables include numerical (integer and float) and categorical
(object) types, covering listing characteristics, host behaviour, location attributes, pricing, and
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 18
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
review scores. While geographic data (latitude, longitude), and basic property attributes
(accommodates, room_type) are complete, critical fields like price (14% missing), bathrooms
(14%) and bedrooms (2%) require imputation. Review-related metrics show about 17% null
values, likely representing new listings without reviews. The dataset contains a mix of numeric
and text data, with some numeric fields like price currently stored as text objects due to currency
symbols. Several date fields need conversion from strings to datetime format, and redundant
columns (multiple variations of minimum/maximum nights) should be consolidated. The
presence of completely empty columns (neighbourhood_group_cleansed, calendar_updated)
which might need to drop. For analysis, the price field will need cleaning to remove currency
symbols. The completeness of geographic data enables robust location-based analysis, while
the variety of host and property metrics provides multiple dimensions for predictive modelling,
though substantial data cleaning will be required first to address missing values and inconsistent
formatting. (See data structure, initial descriptive statistics summary in appendix) The
following histograms provided the key numerical features to understand their distribution and
identify skewness as shown in Figure 3. Most hosts manage few properties, while a small
percentage operate larger portfolios. Property attributes show most listings accommodate 2-4
guests with 1-2 bedrooms/bathrooms, though some extreme values exist. Geographic data
clusters in popular areas, while review scores average 4.7/5, with cleanliness being strongest.
Availability shows a bimodal distribution - either highly available or nearly booked. Short-term
rentals (1-7 days) dominate, though some cater to longer stays.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 19
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 20
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 3: Numerical Data Visualization (Sydney)
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 21
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
3.3.2 Data Cleaning and Data Conversion
As mentioned in initial data exploration, some of the necessary processing are done in this
section. The data cleaning process ensures proper variable encoding for analysis. Price values
with currency symbols are converted to floats, and binary variables like 'host_is_superhost' and
'instant_bookable' are mapped from 't/f' to 1/0. Date columns ('host_since', 'last_review',
'first_review') are converted to datetime, with 'host_since' processed to extract year and month
for tenure and price analysis.
Table 3 (a) Sydney (b) Melbourne missing value summary
As can be seen from Table 3, Missing data, notably in 'host_neighbourhood' (72% Sydney, 76%
Melbourne), is handled by imputing 'unknown' for categorical variables and mode/median for
numeric features like bedrooms, bathrooms, and review scores. Percentage-based fields, such
as host response rates, are cleaned by removing symbols and converting to decimals. Data
quality is maintained through validation checks, including price distribution post-outlier
removal and missing value analysis, with geographic coordinates preserved for location-based
insights.
3.3.3 Outliers
Outlier treatment follows a systematic approach using interquartile range (IQR) methodology
for numeric features. The following Figure 4 illustrates the outlier’s summary of all the Sydney
and Melbourne numeric column.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 22
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 4: Outliers summary (a) Sydney (b) Melbourne
The Figure 5 illustrates the pattern of these outliers in boxplot. The similar pattern can be seen
in Melbourne which has been shown in Appendix. Price values above the 99th percentile are
capped, while bedroom, bathroom, and bed counts are constrained within 1.5 IQR bounds to
mitigate extreme values. The amenities field undergoes special processing, converting the string
representation of lists into actual Python lists for subsequent feature engineering. The outlier
equation is as follows.
Lower Outlier = Q1 – (1.5 * IQR)
Higher Outlier= Q3 + (1.5 * IQR)
IQR = Q3 – Q1
… (1)
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 23
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 24
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 5: Boxplots of Numerical Columns with Outliers (Sydney)
3.3.4 Correlation Analysis
Before further analysis, a correlation analysis has been identified among numerical features.
The correlation analysis shows Sydney's Airbnb prices strongly correlate with property size
(accommodates 0.47, bedrooms 0.47) and coastal proximity (negative correlations with beach
distances), while Melbourne's prices depend more on property features (accommodates 0.42,
bedrooms 0.42) with minimal geographic influence. Both cities show higher review volumes
associate with lower prices.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 25
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 6: Correlation Analysis of (a) Sydney (b) Melbourne
3.3.5 Exploratory Data Analysis
The analysis focuses on 40 key variables with inclusive of extra year column selected for their
demonstrated impact on Airbnb pricing. These features capture essential dimensions including
host characteristics (superhost status, experience, response rates), property attributes (size, type,
amenities), precise location data (coordinates, neighbourhood), booking parameters
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 26
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
(availability, stay rules), and review metrics (quantity and quality). The following Table 4
presents the descriptive statistics of the selected features used in predictive modelling
framework. These metrics provide essential insights into the distribution and variability of key
variables influencing Airbnb listing prices across study areas.
Table 4: Descriptive Statistics of Sydney
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 27
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
There are 7 areas that have been considered in this analysis. The first condition is comparing
price distribution and other attributes. The Price Distribution by Minimum Stay Requirement
(Figure 7 and Appendix) for both cities shows that prices generally increase with longer
minimum stays, peaking around 4-7 nights, then stabilizing or slightly decreasing for 1-2 weeks,
1-2 months, and 1+ year stays, with the median price hovering around 200-300 USD. The Price
Distribution by Maximum Stay diagram indicates a similar upward trend, with prices rising as
the maximum stay length increases, reaching the highest median around 3-12 months and 1+
year stays, also around 200-300 USD.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 28
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 7: Price Distribution (Sydney)
The second analysis is price distribution plots for room types, property types, neighbourhood
status, and host neighbourhoods in Sydney Figure 7 and Melbourne in appendix section reveal
significant outliers across all categories, with prices occasionally spiking above $2000 in
Sydney, particularly for entire home/apartment listings and premium areas like Bondi Beach,
and exceeding $1000 in Melbourne, especially in the Central Business District.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 29
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 8: Average Listing Price by Host Join Year and Month (a) Sydney and (b)
Melbourne
The third analysis is comparing average listing price by host join year and month. The Figure
8 illustrates the trend of average listing prices in Sydney and Melbourne based on the year and
month hosts joined. As it can be seen that the depicting prices by host join year from 2010 to
2024, shows a significant peak at $350 in 2011, dropping to $200 by 2024, while Melbourne
sharpened at $240 in 2020. However, both cities have a general downward trajectory with
fluctuation throughout the years. Both cities have the highest price listing in July with the
fluctuation between $250 and $285 in Sydney and $170 and $200 in Melbourne.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 30
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 9: Geographic Distribution by Room Type (Melbourne and Sydney)
The next analysis is the distribution of Airbnb by geographical map by its room type. Both
cities exhibit a high density of private rooms and entire homes/apartment with around 77.4% in
their central areas with fewer hotel and shared room options with less than 1%. Therefore, it
can say that they have more preferences for these room types in urban cores.
Another concern about the Airbnb analysis is amenities which most people has raised in
awareness of smoke alarm as part of their safeness. This is supported by Kennedy et al. (2025)
who mentioned that hosts have shared detailed experiences of how fires in their Airbnb
properties have affected their lives, as well as emphasizing their broader implications for fire
safety. Another most contribution amenities are Kitchen, Wi-fi, hot water, hair dryer and
cooking utensils which affect the guest experience because these are all home benefit amenities
so that the guests can feel like living in their home even though they are staying in Airbnb.
These factors have been suggested by Tafesse and Tariq (2025) which can influence booking
decisions as guests increasingly prioritize properties that offer a seamless and familiar living
environment.
Figure 10: Top 20 most common amenities (a) Sydney (b) Melbourne
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 31
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 11: Word Cloud of Sydney and Melbourne (Amenities)
Another analysis is listing names for properties using word cloud Figure 12 which highlights
the prominence of apartments and studios as the most common property types, reflecting a
strong urban housing market. The prominence of Sydney and Melbourne, along with notable
suburbs such as Bondi Beach, Manly, St Kilda, and Southbank, highlights a concentration on
these significant areas. Attractive attributes like scenic views, balconies, ample space,
contemporary designs, chic styles, comfort, and luxury are repeatedly featured, indicating their
appeal to prospective tenants or purchasers. Research by Milunovich and Nasrabadi (2025)
supports this, noting that high-priced listings prioritize luxury and lifestyle, with standout terms
like ‘beach’, ‘Bondi’, ‘house’, and ‘view’. Conversely, lower-priced listings emphasize
utilitarian features, with words such as ‘room’, ‘location’, ‘minute’, and ‘station’ reflecting a
more functional approach.
Both cities exhibit a tiered market structure, with entire units commanding higher medians (e.g.,
$145.00-$204.50) than private rooms ($34.50-$100.00), and both have seen steady price
increases over 16 years, driven by inflation and demand. According to KPMG (2025), house
prices in Australia will rise nationally by 3.3% over the next 12 months with a stronger 6% in
2026 in most of the capital cities. The box plot and table data on Australian property prices
(2009-2025) suggest strong demand for entire rental units and homes (medians 300-400 AUD)
and serviced condos/townhouses (200-300 AUD) in Sydney and Melbourne, while private
rooms and cottages (medians below 200 AUD) cater to budget renters. Nationally, average and
median prices rose from $132.23 and $115.00 (2009) to $323.59 and $204.50 (2025), with
surges in 2020-2022. EY UK (2024) denoted that only 68% of households saved less than
£1,500 which means that most of the people can afford for low budget renting based on their
income.
Sydney’s median house price ($1.45M) exceeds Melbourne’s ($980K) due to limited land and
global appeal (Rachwanai, 2023), while Melbourne’s sprawling layout and higher housing
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 32
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
completions (9.5 vs. 6 per 1,000 in NSW) offer affordability. Sydney shows higher price
variability (max >$1,000 AUD) for premium properties, while Melbourne’s post-pandemic
growth lags, reflecting a slower recovery (Wear, 2023). Higher NSW costs and taxes (50% vs.
37% in Victoria) reinforce Sydney’s premium market versus Melbourne’s value-driven one
(Martin, 2024).
According to Airbtics (2025) data, a typical short-term rental in Sydney is reserved for 281
nights and 245 nights in Melbourne annually, achieving a medium Airbnb occupancy rate of
77% and 67%, and an average daily rate (ADR) of AU$241 ($159) and AU$212 ($140). From
June 2024 to May 2025, the average yearly revenue for short-term rentals in Sydney reached
AU$65K ($43K) in Sydney and AU$51K ($34K) in Melbourne. As of July 4, 2025, Sydney
hosts 14,092 and Melbourne hosts 23,975 active Airbnb listings, with October and November
identified as the peak seasonality months.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 33
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 12: Price Distribution by Property Types
Figure 13: Superhost Classifier Feature Importance
Both cities are showing that the guests are choosing Airbnb room and property based on the
review scores rating with an importance level of more than 0.5. However, the host acceptance
rate is also at the moderately important factor with a value of around 0.3 contrasting with host
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 34
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
response rate, which has the least influence. Therefore, in the previous study of Gunter (2018)
who advised Airbnb to focus mainly on maintaining excellent ratings that they are having as
sufficient Airbnb demand criterion 1 and excellent rating as 2 overall whilst the other remaining
factors also should not be neglected.
3.4 Modelling
When examining the machine learning algorithms, there are three types as Supervised,
Unsupervised and Reinforcement learning denoted by Venunagender (2024). This study applies
machine learning to analyze factors affecting the Airbnb market in Melbourne and Sydney.
Eight models were tested on data subsets: Random Forest, XGBoost, Gradient Boosting, K-
Nearest Neighbors, Multiple Linear Regression, Lasso Regression, Support Vector Regression,
and Decision Tree. Five top-performing models were selected for further analysis, using a fixed
random seed of 42 for reproducibility, as supported by Alves (2024). Based on these five
models, cross validation (CV) was used to prevent overfitting and make the models
generalisable. Five folds CV were used in this case as it is a very common procedure. The
method is supported by Alizadeh et al. (2018); Vogtlander (2023); Alves (2024). One study
was used 10 folds CV, and another one was used 3 folds CV. Since this study aligns with
previous studies, RandomizedForestCV was used in Gradient Boosting, Randon Forest and
XGBoost, and GridSearchCV was used in KNN. RandomizedSearchCV reduces computational
expenses by sampling hyperparameters randomly instead of exhaustively searching all
combinations like GridSearchCV does, requiring significantly fewer model evaluations. This is
supported by Chica-Olmo. et al (2020) and Vogtlander (2023) whose studies confirmed
RandomizedSearchCV’s ability to achieve comparable model performance with significantly
reduced training time.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 35
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 14: Machine Learning Algorithms (Venuagender, 2024)
The analysis employs a basic linear regression where the response variable Y is modelled as
the sum of its mean  and random error ϵ.
=+ϵ … (2)
Where  = [μ𝒾], 𝒾 = 1, . . . , n and n is the number of observations (Zhu, Li and Xie, 2020).
The following Table 5 described the models and their key characteristics with mathematical
formulation and diagram.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 36
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Table 5: Comparison of machine learning models used for price prediction
All models were implemented with scikit-learn (Python) using standardized features. Multiple
Linear Regression serves as baseline, while tree-based ensembles (RF, XGBoost) and
regularized methods (Lasso, SVR) address different aspects of the prediction task. Before
accessing the modelling, train-split function was used to split the dataset into training (80%)
and testing (20%). The models were trained and tested on a subset of 10 selected features. The
feature selection was employed due to its ability to identify the most impactful variables such
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 37
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
as bedrooms, longitude, latitude, bathrooms, accommodates, review_scores_location, beds,
availability_365, host_total_listings_count and reviews_per_month. These variables were
selected based on the evaluation of VIF results (Figure 15), feature selection outcomes, and
correlation analysis with price, ensuring the retention of the most impactful variables while
mitigating multicollinearity. This approach is consistent with the current study’s initial VIF
analysis, which identified multicollinearity in housing size and type variables (VIF > 10),
leading to the exclusion of medium-size housing and apartment ratios. The methodology aligns
with Garha and Azevedo (2021), who addressed similar issues by dropping highly correlated
variables (e.g., native in Barcelona, singleph in Lisbon) when VIF exceeded 3, improving model
fit as evidenced by enhanced Akaike Information Criterion and Bayesian Information Criterion,
and Log-Likelihood values. The observed variability in Seoul’s housing market, median price
of 5.34 thousand dollars per square meter and an average of 8.76 Airbnb units per census block
group (Ki & Lee, 2019).
Figure 15: VIF check (a) Sydney (b) Melbourne
Following Phung’s (2024) recommendation, numerical features were normalized using
Standardscaler to achieve a mean of 0 and a standard deviation of 1, ensuring consistent scaling
for models sensitive to feature magnitudes. Both the input features (X) and the target variable
(y) were standardized to maintain uniformity. Performance metrics, including R-squared, MSE,
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 38
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
RMSE, MAE and MAPE were calculated for both training and test sets with 5-fold cross-
validation providing additional robustness.
3.4.1 Principal Component Analysis (PCA) and K-means Clustering
This study employs further two unsupervised machine learning models as PCA and clustering
model to identify the distinct behavioural host groups and market segments of Airbnb. PCA
serves as a dimensionality reduction tool which effectively transform numerous correlated
housing variables into a more manageable set of uncorrelated components while retaining
maximum explanatory power (Del Giudice et al., 2024). This technique proves particularly
valuable for identifying latent relationships within complex real estate datasets of Italian cities.
K-means clustering, on the other hand partitions into distinct groups based on feature similarity.
The clustering process incorporates the Within-Cluster Sum of Squares criterion to optimize
group formation, ensuring meaningful segmentation of rental properties (Del Giudice et al.,
2024). In their analysis, Rome dominates in absolute listing volume (1,259,649 dwellings),
consistent with its larger housing stock, while Milan shows the highest relative platform
penetration (3% of total housing units). Short-term rental prices peak in Rome, contrasting with
long-term markets where Milan commands premium rates. Monocentric cities (Milan, Bologna)
exhibit classical urban economic patterns with clear centre-periphery price gradients and Naples
displays inverted spatial pricing due to its polycentric tourist attractions. Therefore, price
sensitivity appears more pronounced in short-term versus long-term rental markets. These
findings highlighted the combination of PCA, and clustering techniques has particularly
effective in uncovering these nuanced relationships, demonstrating the value of multivariate
analysis for urban housing research. Therefore, the methodology in this study successfully
captures both the expected economic patterns and the platform-specific deviations that
characterize modern rental markets.
3.5 Evaluation
To assess accuracy of the model performance, this study adopted Mean Absolute Percentage
Error (MAPE), Mean Squared Error (MSE), Mean Absolute Error (MAE), R2 scores and Root
Mean Squared Error (RMSE) with cross validation to ensure robustness. Their respective
formulas are as follows:
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 39
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
… (3)
… (4)
… (5)
… (6)
… (7)
… (8)
3.6 Deployment
As for the deployment, it involves integrating the previous mentioned predictive models into a
decision-support system using interpretable AI by comparing local and global interpretability
as using SHAP (Shapley Additive exPlanations) and permutation importance. Using these, it
will be able to enhance transparency, enabling a deeper understanding of feature contributions
and improving decision-making processes for Airbnb to optimize guest services (Panahandeh
et al., 2025). As from mathematical point of view, the SHAP values of the feature importance
are computed using:
… (9)
Where i defines the value of the 𝔦th feature. The feature explanation is written as:
… (10)
Permutation Feature Importance (PFI) measures a feature's relevance by evaluating the increase
in model prediction error after randomly shuffling its values. This method works on the
principle that disrupting a key feature’s relationship with the target variable degrades model
performance. By comparing baseline accuracy (e.g., RMSE) with post-shuffled results, PFI
identifies influential features (Garcia, 2023 in Boehmke & Greenwell, 2019).
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 40
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
4.0 Results and Findings
This section presents the results from the previously mentioned methodology, focusing on
model performance analysis, model comparison, and the outcomes of PCA and k-means
clustering.
4.1 Model Performance
4.1.1 Initial Model Performance
The comparative analysis of machine learning model performance across Melbourne and
Sydney reveals distinct geographic patterns in predictive accuracy. XGBoost led in Melbourne
(Test R²: 0.5107, RMSE: 85.09), while Random Forest slightly outperformed in Sydney (Test
R²: 0.4941, RMSE: 164.98). Ensemble methods (XGBoost, Random Forest) consistently
surpassed Gradient Boosting, KNN, and linear models in both cities, with Melbourne showing
greater predictability due to lower error rates. Decision Trees overfit significantly, while
Gradient Boosting ranked intermediately. A comprehensive evaluation of all models’
performance is presented in Table 6 which include key metrics such as R2 score, MAE, MSE,
RMSE and MAPE result.
Table 6: Initial Model Performance Result: (a) Sydney (b) Melbourne
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 41
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
4.1.2 Tuned Model Performance
Table 7: Tuned Model Performance Result: (a) Sydney (b) Melbourne
Following hyper-parameter tuning, XGBoost excelled in both cities, achieving Test R² of
0.4998 in Melbourne (CV R²: 0.4925 ± 0.0196) and 0.5118 in Sydney (CV R²: 0.5289 ± 0.0350).
Gradient Boosting and Random Forest followed closely, but Melbourne models showed better
precision (RMSE: 86.0365, MAE: 52.4431) compared to Sydney (RMSE: 161.0950, MAE:
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 42
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
81.8186). Ensemble methods outperformed KNN and OLS by 20-30% in Test R², with Decision
Trees consistently overfitting in Table 7 and Figure 16.
Figure 16: Model Performance Comparison (Test set metrics): (a) Sydney (b) Melbourne
As shown in Figure 17, the hyperparameter tuning process identified optimal configurations for
each machine learning model while maximizing their predictive performance. The tuned
configurations demonstrate that while all ensemble methods achieved comparable performance
(R-squared 0.543-0.547), XGBoost showed a slight edge with KNN trailing at 0.4935 R-
squared due to its fundamentally different and distance-based approach. The optimized
XGBoost configuration, utilizing 200-300 estimators with max_depth parameters of 5-7 and a
consistent learning_rate of 0.1, effectively captures non-linear relationships and complex
feature interactions within the data. This carefully tuned architecture enhances the model's
ability to generalize while maintaining computational efficiency, resulting in more reliable
predictions for the study's objectives.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 43
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 17: Parameters of each model by GridSearchCV and RandomizedSearchCV: (a)
Sydney (b) Melbourne
4.1.3 K-Fold Cross Validation Results
K-fold cross-validation (cv) was performed after the tuned models were achieved Table 8. The
R-squared fold scores show the result from each fold in the k-fold cv process for each model.
A higher mean indicates a better predictive accuracy while the Standard deviation cv R-squared
column shows the consistency of the model’s performance across folds.
For Sydney, RF model had the highest mean CV R2 score of approximately 0.538 with a
standard deviation of 0.042, demonstrating relatively strong predictive accuracy and
consistency compared to other models. Followed by XGBoost and Gradient Boosting, which
are also performed well while OLS model had the lowest mean score of 0.38. However, the
reverse pattern can be seen in Melbourne that the XGBoost achieved the highest mean R2
(0.4925) with the lowest variability of Std 0.0196 while Gradient Boosting and Random Forest
followed closely.
Table 8: Results of the K-Fold CV across (1) Sydney (2) Melbourne
4.1.4 Learning Curves and Residual Plots
In order to closer look at the models, learning curves and residual plot were observed in the
following figures. Learning curves visualize the model's training progress and generalization
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 44
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
capability across different dataset sizes. The accompanying residual plots provide insight into
error patterns, validating whether key regression assumptions are satisfied in our predictions.
The evaluation of the learning curves and residual plots for both cities have been shown below.
Figure 18: Sydney Learning Curves for all models
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 45
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 19: Melbourne Learning Curves
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 46
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 20: Sydney Residual Plots
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 47
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 21: Melbourne residual plots
4.1.5 Actual and Predicted Values
The scatter plots comparing actual and predicted values of all models across two cities.
XGBoost outperforms others with 0.512 R-squared up to 2500 in Melbourne and 0.500 R-
squared in Sydney while KNN and Multiple Linear Regression have greater scatter and poorer
fit. The key model performance for both cities of test R2 has been shown in the radar diagram
(Figure 22, Figure 23, Figure 24).
Figure 22: Sydney Actual and Predicted values
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 48
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 23: Melbourne Actual and Predicted values
Figure 24: Model Performance Radar Chart: (a) Sydney (b) Melbourne
4.2 Principal Component Analysis (PCA) and K-means clustering
Principal Component Analysis identified consistent patterns across both cities, with PC1
strongly correlated to property size (Sydney: 30.3% variance, Melbourne: 29.8%) through
features like accommodation capacity and bedroom count, while PC2 captured location quality
and availability metrics (See Table 9 and Figure 25 and Figure 26).
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 49
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Table 9: Principal Components Analysis of (a) Sydney and (b) Melbourne
Figure 25: Sydney Clustering
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 50
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 26: Melbourne Clustering
However, the clustering results reveal distinct market structures between Sydney and
Melbourne Table 10. Sydney's listings organize into four clear clusters: professional host
properties (mean 24.72 listings), balanced mid-range options, premium accommodations, and
budget listings showing 1.71 mean reviews/month - significantly higher than other clusters
despite their less central locations. Melbourne presents a more complex five-cluster ecosystem,
including a unique category of extra-large properties (mean 6.56 accommodates) while not in
Sydney, alongside professional hosts with substantially higher portfolio sizes (Cluster 2: 118.18
mean listings). The review patterns differ notably, while Sydney's budget cluster (Cluster 3)
leads with 1.71 reviews/month, Melbourne's cluster becomes lower between cluster 0 and 4,
suggesting subtle variations in traveller behaviour between the cities.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 51
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Table 10: Cluster profiles (a) Sydney (b) Melbourne
4.3 Feature Importance
This study compared global and local interpretability of XAI to reveal the feature importance.
As for prior global XAI, SHAP and Permutation importance were used to compare while the
latter used LIME when interpreting the best model result XGBoost. In the Figure 27, the
analysis in both cities highlighted bedrooms exceeding 0.20 and accommodates are the most
influential features. The specific score and predicted outcome of a model were illustrated in
partial dependence plots. In this case, bedrooms, longitude and latitude were selected randomly
and revealed how changes in these features. Meanwhile in Sydney, bedrooms showed a clear
upward trend while longitude has a sharp increase, and latitude has a fluctuating pattern.
Melbourne, on the other hand, the bedrooms increased from 0 to 3 and the partial dependence
from 180 to 240. Even though longitude varies erratically between 144.9 and 145.3 with peaks,
the latitude decreased with a fluctuating trend.
Figure 27:XGBoost Feature Importance (a)Sydney and (b) Melbourne
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 52
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 28: Partial Dependence Plots (a) Sydney and (b) Melbourne
4.3.1 Global Interpretability XAI analysis
In this section, the critical insights on feature importance will be explained using global and
local interpretability. Figure 29 exhibits the output of the Explainable AI modelling of Airbnb
listings, to comprehend the nature of the global influence patterns. The top explanatory
variables are shown in the SHAP global feature impact exhibit on the left, using red to denote
high-impact features like longitude and blue for lower-impact ones such as beds or bathrooms.
The variations of the predictive contribution of the explanatory variables are reflected in the
permutation importance exhibit on the right, confirming longitude as the top feature across both
cities, followed by moderate contributions from accommodates, bedrooms, latitude,
availability_365, and reviews_per_month.
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 53
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
Figure 29: Global Explanation (a) Sydney and (b) Melbourne
Figure 30: Local Interpretability (LIME) Sydney
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 54
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
For the LIME analysis, Instance 0 was selected for both Sydney and Melbourne due to its
highest prediction error (residual), making it particularly informative for understanding factors
driving property outcome predictions. This observation is critical because it reveals where the
model’s predictions deviate significantly from actual outcomes, highlighting potential
limitations or unmodeled factors such as unique property or local market characteristics.
For Airbnb listings in Sydney, longitude is the most influential feature, followed by
accommodates, bedrooms, latitude, availability_365, and review_scores_location, as shown by
SHAP global feature impact, with LIME analysis revealing local variations in feature influence
strength and direction; for instance, smaller properties (accommodates ≤ 2, bedrooms ≤ 1,
bathrooms ≤ 1) in specific areas (longitude ≤ 151.20, latitude < -33.90) with moderate reviews
adjust predictions downward from a base value of 114.57, suggesting lower desirability or price
in high-error cases. In Melbourne, longitude remains key, but accommodates and bedrooms
rank higher in permutation importance, with availability_365, latitude, and
review_scores_location moderately influential; local analysis shows small properties
(accommodates ≤ 2, bedrooms ≤ 1, bathrooms ≤ 1) with high availability (availability_365 >
327) and good reviews (review_scores_location = 4.96) also shift predictions downward from
a base value of 151.46, indicating potential overestimation in such listings.
Figure 31: Local Interpretability (LIME) Melbourne
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 55
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
5.0 Discussion of Findings
The comparative analysis of Airbnb price prediction models in Melbourne and Sydney reveals
insights into the efficacy of machine learning models and geographic variations in market
predictability. XGBoost consistently outperformed other models in both cities post-
hyperparameter tuning, achieving test R² values of 0.4998 in Melbourne and 0.5118 in Sydney,
underscoring its robustness in capturing non-linear relationships and complex feature
interactions within Airbnb datasets. This aligns with prior studies emphasizing the superiority
of ensemble methods for housing price prediction, such as Bhadurali (2024) and Liu (2021).
Ensemble techniques, including XGBoost, Random Forest, and Gradient Boosting,
outperformed simpler models like KNN and Multiple Linear Regression by 20-30% in R² scores,
reinforcing their suitability for complex datasets. Notably, Melbourne exhibited greater
predictability, with lower error metrics (RMSE: 86.0365, MAE: 52.4431) compared to Sydney
(RMSE: 161.0950, MAE: 81.8186), likely due to its compact urban layout and centralized
attractions creating more consistent price patterns, as supported by Alexander (2018). In
contrast, Sydney’s sprawling geography, with demand fragmented across coastal areas,
business districts, and suburban hubs, introduces higher variability, making prices harder to
predict uniformly (Milunovich & Nasrabadi, 2025). Proximity to strategic landmarks and the
density of points of interest further emerged as critical pricing determinants in both cities,
consistent with Panahandeh et al. (2025). These findings extend prior research on Airbnb
pricing in Australian cities, which often focused on single-city contexts without direct cross-
market comparisons (Cai et al., 2019; Caminal, 2020; Aum Global., 2025).
A key distinction between Melbourne and Sydney lies in their market predictability and spatial
distribution of listings, which are influenced by urban structure and tourism patterns.
Melbourne’s compact layout and centralized attractions, such as the CBD and cultural precincts,
foster a more uniform pricing environment, contributing to lower error metrics and a higher
degree of predictability. This may be driven by a more regulated tourism framework and a
balanced distribution of amenities, as suggested by Alexander (2018). Conversely, Sydney’s
sprawling geography, with its fragmented demand across coastal tourist zones (e.g., Bondi) and
business districts, results in greater price variability and higher prediction errors, likely
exacerbated by diverse traveller preferences and less centralized regulatory oversight
(Milunovich & Nasrabadi, 2025). These differences underscore the role of urban planning and
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 56
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
tourism policy in shaping Airbnb market dynamics, offering a foundation for tailored strategies
in each city.
Geospatial and property-related features emerged as dominant influencers, as revealed by
SHAP, permutation importance, and LIME analyses. In both cities, variables such as bedrooms,
accommodates, longitude, and latitude exhibited the highest impact, with partial dependence
plots illustrating non-linear trends, for instance, a sharp upward trajectory for bedrooms in
Sydney and erratic variations in longitude in Melbourne. This emphasizes the critical role of
location in pricing, where proximity to urban centres or attractions can drive premiums,
consistent with the clustering results that delineated distinct market segments: four clusters in
Sydney (e.g., professional hosts with high portfolios and budget options with elevated review
rates) and five in Melbourne (including a unique extra-large property cluster). Research by
Panahandeh et al. (2025) indicates that properties closer to city centers have higher Shapley
values, reflecting greater price deviation from the average, which decreases with distance.
Ghosh et al. (2023) suported that total_Listings is the most influential variable, while bedrooms,
bathrooms, and similar features significantly drive prices, as shown by LIME-based analysis.
Local-level feature influence varies in strength and direction, with some features negatively
impacting prices. The Boruta algorithm highlights the importance of including all filtered
features for accurate local predictions. It can be emphasized on monitoring all screened features,
as less prominent ones can significantly enhance prediction accuracy.
PCA further corroborated this, with PC1 explaining around 30% of variance through size-
related features in both locations (Liu, 2022). These patterns reflect subtle differences in
traveller preferences and market structures; for instance, Sydney's budget cluster showing 1.71
reviews per month indicates higher demand for affordable, less central options, while
Melbourne's more fragmented clustering suggests a diverse ecosystem influenced by larger
accommodations (Thackway et al., 2022). Such insights build on prior studies that have
identified location and amenities as key price determinants in Australian Airbnb markets, but
this work advances the field by integrating XAI techniques to provide both global and local
interpretability, revealing how features like high availability (>356 days) or specific coordinates
positively affect predictions in Melbourne (Tang et al., 2023).
The cross-validation results and learning curves reinforce the models' generalizability, with
XGBoost displaying the highest mean CV R² (0.4925 in Melbourne with low std. dev. of 0.0196;
closely followed by Random Forest's 0.538 in Sydney with std. dev. 0.042) and balanced train-
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 57
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
test gaps, indicating minimal overfitting compared to Decision Trees. Residual plots and scatter
comparisons further validate this, showing tighter fits for ensemble methods and greater scatter
for KNN and OLS, which aligns with the radar diagrams depicting test R² hierarchies. These
outcomes highlight the benefits of hyperparameter optimization, e.g., XGBoost's 200-300
estimators, max_depth 5-7, and learning rate 0.1, which enhanced performance across datasets,
achieving R² values of 0.543-0.547 for ensembles versus KNN's 0.4935 (Rezazadeh Kalehbasti
et al., 2019; Alves, 2024). In comparison to existing literature, this study's multi-model
evaluation and geographic focus reveal that while Sydney models exhibit slightly higher
absolute R² post-tuning, Melbourne's lower errors point to inherent market predictability
differences, possibly linked to regulatory or tourism factors not fully explored in prior Sydney-
centric analyses (Soltani & Lee, 2024; Devina & Ardhi, 2025).
From a practical standpoint, these findings have significant implications for Airbnb hosts and
platform stakeholders in Australia. By leveraging XGBoost for price predictions, hosts can
optimize listings based on influential features like bedrooms and location, potentially increasing
revenue through dynamic pricing strategies that account for cluster-specific demands—e.g.,
emphasizing premium amenities in Sydney's high-end segments or capacity in Melbourne's
extra-large clusters. Policymakers could use the clustering insights to monitor market
segmentation and its effects on housing affordability, especially given the spatial impacts noted
in the analyses (Thackway et al., 2022). Moreover, the integration of XAI enhances
transparency, addressing ethical concerns in ML-driven pricing by allowing users to understand
how factors like latitude contribute to outcomes, which is particularly relevant in volatile
markets like Sydney.
6.0 Conclusions and Recommendations
This study applied the CRISP-DM framework to predict Airbnb prices in Melbourne and
Sydney, addressing gaps in multi-city analyses and XAI integration. XGBoost outperformed
other models post-hyperparameter tuning, achieving test R² values of 0.4998 (Melbourne) and
0.5118 (Sydney), with Melbourne’s lower error (RMSE: 86.0365 vs. Sydney’s 161.0950)
indicating greater market predictability. SHAP and LIME analyses highlighted geospatial
(longitude, latitude) and property features (bedrooms, accommodates) as key predictors.
Clustering identified four Sydney segments (e.g., professional hosts with 24.72 listings, budget
options with 1.71 reviews/month) and five in Melbourne (e.g., extra-large properties with 6.56
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 58
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
accommodates), reflecting urban and traveller differences. These findings enhance pricing
transparency and inform data-driven strategies for Australia’s Airbnb market, particularly amid
Sydney’s housing affordability challenges.
Building on the study's insights, several recommendations are proposed for stakeholders to
optimize Airbnb operations and mitigate market impacts. For hosts, leveraging XGBoost
predictions to adjust prices based on high-impact features like location and bedrooms could
enhance revenue, particularly by targeting cluster-specific demands, e.g., emphasizing
premium amenities in Sydney's high-end segments or capacity in Melbourne's extra-large
clusters (Thackway et al., 2022; Gao, 2025). Platforms like Airbnb should integrate XAI tools,
such as SHAP and LIME, for transparent pricing recommendations, fostering trust and
competitiveness, as suggested by studies advocating for interpretable AI in pricing models
(Panahandeh et al., 2025; Tang et al., 2023). Policymakers are advised to implement region-
specific regulations, such as caps on listing density in high-demand areas like Sydney’s coastal
suburbs or Melbourne’s inner-city precincts, to address gentrification and housing shortages
without overly restricting tourism benefits (Rabiei-Dastjerdi et al., 2020, 2022). These could
include monitoring professional host clusters, which account for 37% of Sydney listings, to
ensure balanced market segmentation, aligning with regulatory frameworks proposed by
Milunovich and Nasrabadi (2025). Additionally, guidelines promoting transparency in guest
feedback and superhost status could protect consumer interests and maintain market integrity
(Gibbs et al., 2018). Investors should prioritize properties in predictable segments, such as
Melbourne’s budget or mid-range clusters, using data-driven models to assess risks in volatile
Sydney markets and capitalize on occupancy trends (e.g., 77% in Sydney vs. 67% in Melbourne)
(Airbtics, 2025; Pittala et al., 2024). Future urban planning could incorporate these models to
promote sustainable tourism, balancing short-term rental growth with long-term housing
affordability through incentives for conversions back to residential use, as supported by studies
on urban housing pressures (Liao & Nawaz, 2024; KPMG, 2025). These actions could help
alleviate Australia’s rental crisis while supporting economic contributions from the sector.
7.0 Limitations and Future Research
Despite its contributions, this study faces several limitations that warrant consideration. The
reliance on static datasets from Inside Airbnb limits the capture of real-time fluctuations, such
as seasonal demand, potentially reducing generalizability to dynamic market conditions. This
May Myo Khin (MSc. Business Analytics, Queen’s Business School) 59
Airbnb Market Investigations: A Comparative Study of Pricing, Host Behaviour And Business Models in Melbourne and Sydney
finding was supported by Ayyasamy et al. (2022) who stated a lack of analytical forecasting
tools for real time price changes and market demand. Data itself does not provide any external
factor to consider the geographic scope, confined to Melbourne and Sydney, overlooks rural or
other urban Australian contexts, where Airbnb impacts on housing may differ due to lower
tourism density. Another limit that concerns about is the real price value in the dataset as there
are many unusual prices that exceeds than the real price. This case has been noted by Chen and
Xie (2017) that it is valuable to understand the market information given by the host. Potential
overfitting in ensemble models, particularly Decision Trees, highlights the need for more robust
validation across diverse datasets. Future research should incorporate time-series models for
dynamic forecasting, expand to multimodal inputs (e.g., text from descriptions or images), and
employ causal inference to evaluate regulatory impacts, such as short-term rental caps, on
housing affordability. Extending the analysis to additional Australian cities or integrating
emerging AI techniques, such as advanced neural networks, could further enhance predictive
accuracy and policy relevance in Australia’s evolving short-term rental landscape for better
solution.
