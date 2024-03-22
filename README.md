# Ukraine War Twitter Discourse Project

This repo is about exploring how Events happening during the Ukraine War effected Discourse on Twitter. We aim to create a reproducable justifiable method of analysis with professional presentation. We model count data of relevant tweets to a particular article wih a peicewise exponential decay model, then interpret and compare latent variables of events based off topic.


## Outline and Deliverables

(* indicates a section in choices discussing what and why descisions were made)

Data Loading Steps
 
 - [x] Download Data Convert to feather (download_data.ipynb)
 - [x] Convert Washington Post mhtml file to tabular data set (process_newmhtml.ipynb)
 
First Output Steps

- [x] Intake data with Dask (*Library Choices)
- [x] Create function to filter tweets from twitter dataset between 3 days plus or minus the original posting of each article
- [x] Filter dataframe by tweets relevant to each article groupby date and hour (*Relevance Determination Justifications)
- [x] Correct oscilatting patterns in data samples (*Data Correction Justifications) 
- [x] Fit model, filter models by goodness of fit (*Model Details)
- [ ] Report latent fits and create visuals

Cleaning Up

- [ ] Move model definition to its own file
- [ ] Remove %run macros
- [ ] Convert notebooks to Scripts
- [ ] Ensure reproducable pipeline for obtaining results
- [ ]  Run output steps on all articles on cluster

Final Report

- [ ] Give clear interpretations of model latent variables
- [ ] Show visual example of model fits
- [ ] Cluster articles by topic and show relations to latent variables



## Choices

### Library Choices

For this project, we needed a framework capable of large scale data analysis and query. The original dataset has over 2 million tweets and is distirbuted across multiple pickled csv files corrosponding to individual days. We opted for a map reduce approach deciding on Dask over PySpark. We chose Dask as we planned to stay in the Sklearn Scipy ecosystem. For our probabilistic modeling we chose numpyro over pyro, stan, and pymc. As of the writing of this article, Stan and PyMC either have dependencies or themselves do not have wheels for Python 3.11. The model that we use is Peicewise and we wanted Markov Chain Monte Carlo methods that were gradient free as the gradient is not continous. Pyro does have HMC samplers which were capable of fitting the data but NumPyro was both faster and lighter at fitting models. We planned to use Scipy Optimize as a last resort.


### Relevance Determination Justifications

Our approach was always to give relevance scores based off cosine simularity of vectorized representations between each tweet and article. We chose to use a TF-IDF instead of a more SOTA from hugging-face or BERT vectorizer because we preferred performance over accuracy. Our model assumes normal errors and reports standard deviation of predictions making the fuzzy results of this stage acceptable. We chose to bin data by hour. Instead of taking the mean of the relevance scores, we chose an approach that would rid oscillatory bias while maintaining the unit dimesnionality. Dividing the sum of the relevance score over the total count of tweets for the bin would give us the rate of relevance per tweet for the hour. We wanted to model count of relevant tweets by hour.

We face two problems with this approach. We need to find a threshold for determining whether a tweet is relevant or not based on its score, and we need to normalize the data inorder to ensure our model will fit well.

The threshold is based somewhat arbitrarily on if the tweet is in the top 95% of tweets that are relevant. Any cutoff would be somewhat arbitrary, even the rate of relevance itself would be. However, this approach has the benefit of creating large distinct change points in the timeseries. 
 
The second issue is that there is a clear bias in most US based social media services, engagement is higher at certain hours of the day. This creates a clear oscilatory trend in the relevant tweet count. The main issue is that our model is not equipped to handle such a trend, and accounting for it would increase complexity. Thus, we adjust for the hourly trends by the following rational

Given that there were x amount of relevant tweets in hour h, what is the expectation of the relevant tweet count if it occured in all other hours.

All this means is that for the time series grouped by hour (not binned), we figure out the scaling factor that if multiplied by that hours count would make it equal to the average count. We then apply that scaler to the entire time series based on the hour of the observed unscaled relevant tweet counts.  Adjusting for too many trends like day of week wise or day of week and hour wise was unnecessary after this adjustment, and might also overfit. The purpose of this scaling is solely to make it easier to fit model.

### Model Details

The model we use is of form

$` \beta_0 `$ : Baseline before change point

$` \alpha `$ : Relevance Decay

$` \phi `$: Relevance Bias

$` \beta_1 `$ : Baseline after change point

$` \sigma^2 `$ : Variance of predictions

$` t_c `$: Change Point

$` x_t \sim N(\beta_0, \sigma^2) \; \forall t < t_c `$

$`x_t \sim N(e^{- \alpha (t - t_c) + \phi} + \beta_1, \sigma^2) \; \forall t >= t_c `$ 

The model does not account for events with multiple peaks, but is simple enough to interpret and fit easily. 
