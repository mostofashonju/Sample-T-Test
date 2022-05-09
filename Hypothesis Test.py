#!/usr/bin/env python
# coding: utf-8

# # A one-sample t-test checks whether a sample mean differs from the population mean. Let's 
# create some dummy age data for the population of voters in the entire country and a sample of voters 
# in Minnesota and test the whether the average age of voters Minnesota differs from the population:

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import math


# In[9]:


np.random.seed(6)

population_ages1 = stats.poisson.rvs(loc=18, mu=35, size=150000)
population_ages2 = stats.poisson.rvs(loc=18, mu=10, size=100000)
population_ages = np.concatenate((population_ages1, population_ages2))

minnesota_ages1 = stats.poisson.rvs(loc=18, mu=30, size=30)
minnesota_ages2 = stats.poisson.rvs(loc=18, mu=10, size=20)
minnesota_ages = np.concatenate((minnesota_ages1, minnesota_ages2))

print( population_ages.mean() )
print( minnesota_ages.mean() )


# # Notice that we used a slightly different combination of distributions to generate the sample data for Minnesota, 
# so we know that the two means are different. Let's conduct a t-test at a 95% confidence level and see if it correctly 
# rejects the null hypothesis that the sample comes from the same distribution as the population. To conduct a one sample 
# t-test, we can the stats.ttest_1samp() function:

# In[5]:


stats.ttest_1samp(a = minnesota_ages,               # Sample data
                 popmean = population_ages.mean())  # Pop mean


# In[6]:


stats.t.ppf(q=0.025,  # Quantile to check (Top tail 2.5% and bottom tail 2.5%) 5% level of significance
            df=49)  # Degrees of freedom   [df = N-1]


# In[7]:


stats.t.ppf(q=0.975,  # Quantile to check
            df=49)  # Degrees of freedom [df = N-1]


# In[ ]:


#if we that calculated value statistic=-2.57 is less than -2.0095 so we reject the null hypothesis

#check p-value
#if p-value is less than 0.05,null hypothesis rejected....

#We can calculate the chances of seeing a result as extreme as the one we observed (known as the p-value) by passing the 
#t-statistic in as the quantile to the stats.t.cdf() function:


# In[8]:


stats.t.cdf(x= -2.5742,      # T-test statistic
               df= 49) * 2   # Multiply by two for two tailed test *


# # Note: The alternative hypothesis we are checking is whether the sample mean differs (is not equal to) the population mean. 
# Since the sample could differ in either the positive or negative direction we multiply the by two.
# Notice this value is the same as the p-value listed in the original t-test output. A p-value of 0.01311 means we'd expect to see data as extreme as our sample due to chance about 1.3% of the time if the null hypothesis was true. In this case, the p-value is lower than our significance level α (equal to 1-conf.level or 0.05) so we should reject the null hypothesis. If we were to construct a 95% confidence interval for the sample it would not capture population mean of 43:

# In[10]:


sigma = minnesota_ages.std()/math.sqrt(50)  # Sample stdev/sample size

stats.t.interval(0.95,                        # Confidence level
                 df = 49,                     # Degrees of freedom
                 loc = minnesota_ages.mean(), # Sample mean
                 scale= sigma)                # Standard dev estimate       #check with 95% confidence intervals


# # On the other hand, since there is a 1.3% chance of seeing a result this extreme due to chance, it is not significant at the 
# 99% confidence level. This means if we were to construct a 99% confidence interval, it would capture the population mean:

# In[11]:


stats.t.interval(alpha = 0.99,                # Confidence level
                 df = 49,                     # Degrees of freedom
                 loc = minnesota_ages.mean(), # Sample mean
                 scale= sigma)                # Standard dev estimate       #check with 99% confidence intervals


# # With a higher confidence level, we construct a wider confidence interval and increase the chances that it captures to true mean
# ,thus making it less likely that we'll reject the null hypothesis. In this case, the p-value of 0.013 is greater than our 
# significance level of 0.01 and we fail to reject the null hypothesis.
