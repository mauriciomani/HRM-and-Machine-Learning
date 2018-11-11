# Human resource management and machine learning research: My experience
----

#Translations:
* [English](README.md)
* [Español](README-es.md)

<br/>
<p align="left">
<img src="img/iit.JPG" alt="iit">
</p><br/>

## Introduction
This article is divided in two parts. The first is my experience developing a Systematic Literature Review in the Indian Institute of Technology, Delhi. The second is what I found important in the human resource field regarding artificial intelligence. And I provide an example using a Kaggle dateset for the attrition risk in any enterprise (if an employee is leaving or not). 

## Indian Institute of Technology Delhi
Being in India has been one of the best experiences I have ever had. I am very proud of being part and made so great friends in one of the most prestigious universities in India: Indian Institute of Technology Delhi, IIT Delhi. Also tell all my new friends about my great university in Mexico and about Mexico in general. 
When I arrived, I realized not everyone was going to speak English in India, almost all people understand (I think so) but speaking for them is difficult. When I arrived to Satpura, what I called house, an almost graduate student, Dhruv, helped me out get my SIM activated and tried to fix out my stay in Satpura, complicated. Later, my friend and right hand, Swapnil got everything done to get my room: SA-10 in the ground floor, way better in the ground, there, it was not that hot. First day was the thing! No chain, very light and uncomfortable mattress and a lot of mosquitoes made me woke up around 5 am. 
The adventure started! The food in the mess was very good, at least for me, as a foreigner I found it great. However, some friends told me, having the same food for four years is hell.  
I did not know anyone in India at that time (except for Prof. Vigneswara, Swapnil and Drhuv), and I did not have any classes. So I met people on the mess or on the TV room. One of the very first friend I made in India was Jeevarej from Tamil Nadu, doing his postdoc in mathematics, working in fuzzy logic (later I realized how important is for modeling). I just asked him, "do you speak English?" He told me he does, and we started speaking, he also told me that he does not speak Hindi so when he arrived to IIT Delhi, he was asking the same question. 
Delhi is very beautiful, so diverse and full of heritage, is in the heart of India, in fact that is what Delh means in English: HEART. If you are there you must visit AGRA and JAIPUR: THE GOLDEN TRIANGLE. 
<br/>
<p align="center">
<img src="img/dhruv.JPG" alt="dhruv">
</p><br/>
<br/>
<p align="center">
<img src="img/jeev.JPG" alt="jeev">
</p><br/>

## India culture.
I enjoyed the movies, music, food and art. India is full of vegetarians, nevertheless, you can have chicken. I had chicken masala, pani puri, Palak Paneer, Malai Kofta, Kaali Daal, Chole Bhature, Aloo Ka Halwa, Paratha, Sweet Lassi, Curry, Chicken Biriyani, Jalebi, Gulab Jamun, Idili, Melu Vada, Masala Dosa, Tandori Chicken, Chapati. 
<br/>
<p align="center">
<img src="img/cur_pp.png" alt="food">
</p><br/>
Movies are full of emotions. Everyone studying in an engineering college in India have watched "3 IDIOTS" is a great movie, so real! PK, Barfi!, Namastey London, were very good movies as well. I went to a Bollywood musical: ZANGOORA, in Kingdom of dreams. 
I enjoyed visiting Delhi, Agra and Jaipur, their temples, their markets and all sites. 
I visited a lot of places, from the very famous Taj Mahal in Agra to a Sunday old books market in Delhi.
<br/>
<p align="center">
<img src="img/qutab_minar_book.png" alt = "qutab_minar_book">
</p><br/>


## Decision Rules in a HRM context
After reading around 150 papers I discovered two important things, there are no clear definitions regarding data mining, data science, machine learning and so on; secondly, the importance of understanding algorithms, and the use of decision rules.
Data Science is a broad discipline, even though the concept is recent, every day is evolving. 
>According to Berkeley School of Information, the Data Science Life Cycle has five stages, this stages are not exclusive from one another. This five stages are Data Capture, Data Maintain, Data Process, Data Analysis and Data Communication. 

The latter is the most important activity in businesses. It is where we deliver Data Visualizations, Data Reports, Business Intelligence and Decision Making.  Most often, all the time spent in the remaining stages will end up in making decisions, based on the whole process, so we can constantly improve. 
When trying to solve a data science problem, there are plenty of techniques you can use in order to puzzle it out; for example, support vector machines (SVM), decision trees, logistic regression, neural networks and many others. The big problem is that almost all of them are black boxes, you probably know what the algorithm is doing, hopefully, but your co-workers or random ordinary people might not understand the intuition behind those complex models. So as a data scientist you should not only extract value from data but also being able to translate results into solutions and then communicate that. Decision rules will help us extract clear conclusions on how the algorithm is making decisions. So we can later on make decision based on that. 
We can extract decision rules using Python and some well-known libraries.  We are going to use an HR dataset provided by Kaggle contest. The features of the dataset are satisfaction level, last evaluation, number project, average monthly hours, time spend company, Work accident, left, promotion last 5 years, sales, and salary. <strong> Left </strong> variable will be our target. Basically we want two extract two things from our algorithm, the prediction of our target variable, in this case whether the employee leaved or not and why they are leaving. What features increases attrition risk? Those features will be the rest of the dataset, the variables that we will use to feed the decision tree. Basically, the root node is our entire dataset that will later be splitted based on our selected strategy like Gini Index, Chi-square, entropy or variance reduction. The dataset has 10 columns and 14,999 observations. 2 features are object data types, to work with scikit-learn we will transform to dummies those features, using pandas. We will use a train and a test data; the test will have 35% of the whole data. 
To make the tree easier to interpret we are going to set the maximum depth of the tree to 5, so we can have more representative samples from all nodes. The score stills being pretty good with .97, and we can generalize the rules provided by the tree more easily. Like in everything, we will face the trade-off, between a bigger precision and an easier to interpret algorithm.  
Once we call all the methods we will get a dot file, we can open that file with a text processor. Do not worry, it is not that complicated. Essentially, is a bunch of steps to build a visual decision tree, you can do that with python as well.  But it is even easier to visualize all that code, just using the web! Copy and paste all the lines to: <br/>http://www.webgraphviz.com/ <br/>Then, just click on <strong>Generate graph</strong>.
<br/>
<p align="left">
<img src="img/webgraph.PNG" alt="webgraph">
</p>
You will automatically get a decision tree. If it is to robust, probably you might consider moving the parameters of the algorithm. 

## Ready?
<br/>
<p align="center">
<img src="img/decision_tree.PNG" alt="dec_tree">
</p>
Lets extract the rules from the decision tree. First some concepts:<br/>

* Entropy: 
Measure of unpredictability of information content in other words is how much information we learn on average from one instance. We are looking for zero when speaking about entropy. Zero means there is only one label in the node. 

* Samples:
The amount of observations in each node. 

* Value:
The amount of observations in each label. The one on the left is the zero value and the one on the right is the number 1. In this case  in and left respectively. 

To extract decision rules from our tree we must consider the nodes at the end (end nodes) and then all the way up until the root node (backward) or vice versa (forward). It is important to cover all the branch. Otherwise our decisions will be less accurate. 
To extract relevant conclusions, we must ponder those end nodes with tons of observations no matter they are 0 or 1 (in or left). The most relevant include 1039, 981, 5261, 631 samples.<br/>
1. If average month worked hours are more than 126 hours and the result on the last evaluation is between 0.445 and 0.574, and he or she  has less than 2 projects and a satisfaction level less than 0.465 he or she will be more inclined to leave the organization. 
<br/>
<p align="center">
<img src="img/dec_1039.PNG" alt="dec_tree">
</p>
2. If satisfaction level of the workers is less than 0.465 and the average monthly worked hours are less than 290 hours and finally the  number of projects is between 3 and 6, then the employee will move towards staying in the organization
<br/>
<p align="left">
<img src="img/dec_981_5261.PNG" alt="dec2">
</p>
3. If the worker has between 3 and 5 projects and worked less than 290 hours and has less than 4 years in the organization and a satisfaction level more than 0.465 they will be more inclined to stay in the company. <br/>
4. If the time spent in the organization is between 5 and 7 years and the last evaluation scoring is more than 0.8 points, working more than 214 hours on average a month, having a satisfaction level more than 0.46 they will be prone to leave the company.
<br/>
<p align="right">
<img src="img/dec_631.PNG" alt="dec3">
</p><br/>

Finally, Plotting the importance of the features might be a good idea. This can give us a clue on what a worker values the most or at least worries the most. You can obtain this information through the feature_importances_ parameter. And then you just have to plot it. 
Definitely satisfaction level is very importan when making a decision, also the years in the company and the last evaluation, last but not least the number of projects and the monthly hours working. We can say that accidents in the organization, promotions, salary and position are irrelevant. 
<br/>
<p align="center">
<img src="img/importance_of_the_features.png" alt="i_f">
</p>

## Conclusions
Hiring an employee is very costly and firing and employee it is even more costly (you have to take into account all the hiring costs as well). All the rules extracted using this dataset might not be considered to make decisions in an enterprise. This kaggle dataset is simulated but is very useful to visualize how to extract information from enterprise data and convert it to real policies. This data provided by Kaggle is almost clean, so that process was completely easy. When working in a real data base (specially in HRM) your data can be very messy, so be aware of that, you might spend around 80% of your time cleaning the data. Do not be scared of simple statistics they are always usefull and easy to understand for everyone. Always remember knowing why you are picking a particular decision is very important, that is why we are using deicision rules. Once we know why they are leaving we can motivate them accordingly, we are leaving in a tailored world. 
