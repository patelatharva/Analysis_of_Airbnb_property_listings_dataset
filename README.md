# Analysis of Airbnb dataset on property listings
Airbnb published dataset related to its property listings in Seattle and Boston from 2016. 
The dataset for each city contained following files:
* `listings.csv`  contains details about property listing like location, neighbourhood, amenities, number of beds and bathrooms, average ratings etc.
* `calendar.csv` contains details about the availability and price of each property listing on different days of the year 2016.
* `reviews.csv` contains reviews for property listings by users

I followed the steps recommended in [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) methodology to explore this dataset, perform necessary transformations, extracting features and prepare statistical model from it to estimate the booking price of the property on Airbnb.

You see my [project notebook](https://patelatharva.github.io/Analysis_of_Airbnb_property_listings_dataset/analysis.html) containing the steps that I performed.

I asked and found answers for a set of questions through my exploratory analysis of the data related to Seattle property listings on Airbnb in 2016.

### How does the trend in number of properties becoming available over the year look like?

![](https://cdn-images-1.medium.com/max/800/1*tgA_bbkS0TfcqMi3CBA6OA.png)

As we can see in this graph, there was a sharp increase in the properties becoming first time available on Airbnb in Seattle during the early part of 2016 from January to March. After the March, the rise was not so steep, although it the growth continued steadily throughout the year.

### What was the trend in occupancy rate of properties over the year?

This helps us understand what the percentage of listed properties are booked during the year in Seattle in 2016. It can be seen as the rate of utilisation of the listed properties through Airbnb.

![](https://cdn-images-1.medium.com/max/800/1*zWNjEiQu-opB2ZNVTGVV2A.png)

As shown in the graph, the occupancy rate fluctuates significantly between January and March, because there are large number a new properties becoming available first time during that period. Between April to July the occupancy rate is 22–25%. The maximum level of occupancy across all the listed properties was achieved to be just above 30% during the month of July, which is a summer season in Seattle. After July, the occupancy rate slumps from 30% to 23% by end of the year.

### How many properties were occupied during the year?

The number of properties occupied is proportional to number of people choosing to book properties on Airbnb in Seattle in 2016.

![](https://cdn-images-1.medium.com/max/800/1*dc3fVtYmZdqG89FvV0B_dA.png)

As we can see from this graph that there are between 400 to 500 properties occupied during January to April. Starting from April, this number jumps to near 600. In July, there is one more jump with over 700 properties getting occupied. The occupancy drops gradually after July towards the end of year, with nearly 650 properties being occupied at the end of the year 2016.

### What was the trend in average price of property during 2016?

![](https://cdn-images-1.medium.com/max/800/1*gu0zfAiYv0--fabnj5upiQ.png)

As visible in the graph, there is a steady overall rise in the booking price from January to July. In July, the average booking price crosses 150 USD and stays above 150 till end of August.Starting from the September, the average price starts dropping. Near the end of the year during Christmas, the prices rise slightly.

There are roughly 4 spikes in the average price every month, can you guess why? Let’s zoom into the month of July.

### How does the average price change over the days of week in the month of July?

![](https://cdn-images-1.medium.com/max/800/1*45GuSYkOa036ChaZ06JAeA.png)

As we can see, the price tends to be higher on Friday and Saturday than during the rest of the days. May be, it can be attributed to more people choosing to travel or stay through Airbnb listed properties on weekend.

![](https://cdn-images-1.medium.com/max/800/1*bC1ty4VaYKRv-bhV8fePSQ.png)

As we can see, majority of the properties are of type apartment or house. There are also significant number of condominiums, townhouses and lofts.

Coming to the question that I started this post with.

### Can we predict the booking price of the property based on its location, size and amenities?

The dataset has file containing the attributes for the properties listings. It contains attributes including location, neighbourhood, amenities, size, number of bathrooms and beds, property type etc. Our common sense suggests that these attributes play an important role in determining the price of the property for booking on Airbnb. I investigated further in that direction by training a statistical model using Ridge regression technique to see if these attributes can be used to guess the price of a property.

Following the standard process of training a machine learning model, I first prepared the dataset by performing these steps.

1.  Joined two data frames, where one data frame contained the price of the listing on different different days of the year, while another one contained the attributes of that property like location, size, amenities etc. This helped me take into consideration the day of week and month while estimating the price of booking a property.
2.  Eliminated rows with missing values for target variable price
3.  Converted categorical variables into columns with binary numerical values for each categorical level
4.  Filled the missing values of the numerical columns with the mean of the column. I chose not to fill the missing values for categorical variables.
5.  Selected columns from the data frame that would serve as input variable X and target variable y.
6.  Split the X and y datasets into training and test datasets.
7.  Fitted the Ridge regression model on training dataset.
8.  Predicted the prices of properties in test dataset.
9.  Evaluated the model performance by calculating r^2 score between test values and predicted values of prices.
10.  Interpreted the coefficients for each of the input variable in the trained regression model to understand its influence on price prediction.

Here is the list of input variables with highest absolute value of their coefficients in the trained model. They can be thought of as the attributes of the property and time of the year that has maximum significance in predicting the booking price of the property. Positive and negative value in the  `coef`column associated with  `input_variable`  represent the amount of positive and negative influence that input variable has on estimation of the booking price of property respectively.

![](https://cdn-images-1.medium.com/max/800/1*YAkAFatNJAIpA93Oq8gdCw.png)

As you can notice, the top two input variables with largest influence on the estimation of price are related to the type of the property. Another big influencer is the location and neighbourhood of the property. Number of bathrooms in the property is also a significant influencer of price according to this statistical model.

Let’s dig deep into the coefficients of different types of input variables to understand how much influence they have on estimating the price.

#### Influence of type of property on price estimation

![](https://cdn-images-1.medium.com/max/800/1*WckTAajHLrizbIPPOFFFwQ.png)

-   The properties of type Boat are very likely to have higher booking price.
-   The dorm, shared room or tent type properties are very likely to have. lower booking price

#### Influence of neighbourhood of property on price estimation

![](https://cdn-images-1.medium.com/max/800/1*rMr3QG7VU9Tp_MO5twgtiA.png)

As per these table, the properties listed in neighbourhoods including Southeast Magnolia, Windermere, Westlake, Pike Market and Pioneer Square are more likely to have high booking price. Whereas, the properties located in Crown Hill and Laurelhurst are likely to have lower price for booking.

#### Influence of the month of year on price estimation

![](https://cdn-images-1.medium.com/max/800/1*65x-sDVZwnKjNIZ7E6jJOA.png)

In the month of June, July and August; the property is likely to have higher price of booking, whereas in the month of January and February, the property is likely to have lower price of booking.

#### Influence of the day of week on price estimation

![](https://cdn-images-1.medium.com/max/800/1*8-5LpXsSxsrpgAJq7Wvmiw.png)

The property is likely to have higher price on Sataurday and Friday, while it’s likely to have lower price on other days.

#### Influence of amenities on price estimation

![](https://cdn-images-1.medium.com/max/800/1*QOPUWju7Zk5L16uNcqtVFg.png)

Properties with amenities such as availability of doorman, pool, air-conditioner, cable tv, suitability for events, hot tub, gym etc. are very likely to have higher price of booking. Properties allowing to smoke are likely to have lower price of booking.

#### Conclusion

Observing the coefficients of the trained model, there are several conclusions that can be made that I am restating here.

1.  The properties of type Boat are very likely to have higher booking price
2.  The dorm, shared room or tent type properties are very likely to have lower booking price
3.  The precise location and neighbourhood that the property is located play very important role in deciding its booking price.
4.  The number of bathrooms and bedrooms in the property are in strong positive proportion with the booking price.
5.  The prices are also in positive proportion with the number of people it accommodates.
6.  The amenities that strongly and positively influence the booking price are having doorman, pool, cable TV, hot tub, gym and elevator in the building.
7.  The booking price tends to be higher in months July and August, while it tends to be lower in January and February.
8.  Booking prices for Friday and Saturday tends to be higher relative to other days.
9.  Permission of smoking as amenity has strong influence on the price on the negative side.
10.  Properties located in Phinney Ridge Seattle city are less likely to have high price as per this trained model on the available data.
11.  As all the properties in the dataset are located in Seattle, the attributes like market and jurisdiction are not playing any role in determining the price as they are not adding any variance in the input data. However, they might become significant influencers while training this same statistical model on dataset containing listings across multiple cities in the U.S.