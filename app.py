# Naive Bayes Classifier
import streamlit as st
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

im = Image.open("data/favicon.ico")
st.set_page_config(page_title="Naive app",page_icon=im,layout='wide',initial_sidebar_state="expanded")

from model.naive_tutorial import *

hide_menu = """
<style>
#MainMenu {
    visibility:hidden;
    
}

footer{
    visibility:hidden;
}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)




def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://cyberkid.vn/wp-content/uploads/2022/03/01.-ML-la-gi.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    
#set_bg_hack_url()


def homepage():
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.title("Naive Bayes Classifier")
    
    st.markdown("""
                * Naive Bayes is a **supervised learning algorithm**, which is based on **Bayes theorem** and used for solving classification problems. 
                
                * It is mainly used in text classification that includes a high-dimensional training dataset. 
                * Naive Bayes Classifier is one of the simple and most effective classification algorithms which helps in building the fast machine learning models that can make quick predictions.
                * It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.
                * Some popular examples of Naive Bayes Algorithm are **spam filtration, Sentimental analysis, and classifying articles**.
                """)

    left, right = st.columns([1,1])
    with left:
        image = Image.open('data/naive_spammail.png')
        st.image(image, caption='Spam filtration', width=500)
        
    with right: 
        image = Image.open('data/sentiment_exx.png')
        st.image(image, caption='Sentimental analysis', width=400)

def header(url):
     st.markdown(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>', unsafe_allow_html=True)
     
     
def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("data:image/png;base64,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");
             background-attachment: fixed;
             background-size: auto
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

#add_bg_from_url() 


# with open('style.css') as f:
#     st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)


if __name__ == "__main__":
    
    
    #header("notice")
    
    
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.title("Naive Bayes Classifier")
    
    st.markdown("""
                * Naive Bayes is a **supervised learning algorithm**, which is based on **Bayes theorem** and used for solving classification problems. 
                
                * It is mainly used in text classification that includes a high-dimensional training dataset. 
                * Naive Bayes Classifier is one of the simple and most effective classification algorithms which helps in building the fast machine learning models that can make quick predictions.
                * It is a probabilistic classifier, which means it predicts on the basis of the probability of an object.
                * Some popular examples of Naive Bayes Algorithm are **spam filtration, Sentimental analysis, and classifying articles**.
                """)

    left, right = st.columns([1,1])
    with left:
        image = Image.open('data/naive_spammail.png')
        st.image(image, caption='Spam filtration', width=500)
        
    with right: 
        image = Image.open('data/sentiment_exx.png')
        st.image(image, caption='Sentimental analysis', width=400)
        

    left, _ = st.columns([3, 1])
    with left:
        st.write(" ## Why is it called Naive Bayes?")
    
    st.markdown("""
            The naive bayes algorithm is comparised of two words Naive and Bayes. Which can be described as:
            * **Naive**:  It is called Naive because it assumes that the occurrence of a certain feature is independent of the occurrence of other features. 
            Example: if the fruit is identified on the bases of color, shape, and taste, then red, spherical, and sweet fruit is recognized as an apple. Hence each feature individually contributes to identify that it is an apple without depending on each other.
            * **Bayes:** It is called Bayes because it depends on the principle of [Bayes' Theorem](https://www.javatpoint.com/bayes-theorem-in-artifical-intelligence)
                
                
                
                """)
        
    
    # Bayes' Theorem 
    
    left, _ = st.columns([3, 1])
    with left:
        st.write("## Bayes' Theorem")
        
    st.markdown("""
            * Bayes' theorem is also known as **Bayes' Rule** or **Bayes' law**, which is used to determine the probability of a hypothesis with prior knowledge. It depends on the conditional probability.
            * The formula for Bayes' theorem is given as:
                """)
    _,center, _ = st.columns([1, 3, 1])
    
    with center: 
        image = Image.open('data/bayes_theorem.png')
        st.image(image, caption='Bayes Theorem', width=600)
    
    st.markdown("""
    > Where, 
    >> * P(y|X) is called the posterior probability: the probability of the objective y provided that there is a characteristic X
    >> * P(X|y) is called likelihood: probability of feature X given the objective y
    >> * P(y) is called the prior probability of the objective y
    >> * P(X) is called the prior probability of the feature X .

                """)
    
    st.write("Here, X is the vector of features, which can be written as:")
    
    _,center, _ = st.columns([1, 3, 1])
    
    with center: 
        image = Image.open('data/feature_x.png')
        st.image(image, caption='vector of features', width=500)
        
    st.write("Then the Bayesian equality becomes:")
    _,center, _ = st.columns([1, 3, 1])
    with center: 
        image = Image.open('data/pro_x.png')
        st.image(image, caption='Bayesian equality', width=600)
    st.markdown("""
    In the Naive Bayes model, two assumptions are made:
    The features included in the model are independent of each other. That is, changing the value of one feature does not affect the remaining features.
    The features introduced into the model have an equal effect on the target output.  
                """)
    
    st.write("The objective result y for P(y|X) to be maximized becomes:")
    _,center, _ = st.columns([1, 3, 1])
    with center: 
        image = Image.open('data/argmax.png')
        st.image(image, caption='equaliton', width=600)
    
    
    st.markdown("---")    
    # Example
    st.write("## Example")
    st.write("Consider a simple dataset about an employee's lateness to work. The data set is presented in tabular form below:")
    left, right = st.columns([2, 2])
    with left:
        st.write("Suppose, to predict for a day **X=(Muộn, Xấu, Mưa)**, it is necessary to calculate:")
        st.write("By making a frequency table for each feature against the objective, P(X|y) can be calculated.")
        
        st.markdown(f"""
                    <p> <strong>Step 1</strong> We calculate P(X/y) of each feature.<br> 
                    P(X|Muộn) = P(Muộn/Muộn)*P(Xấu/Muộn)*P(Mưa/Muộn)<br> 
                    = ⅗  * ⅖  * ⅕  = 6/125.<br>
                    <br>
                    P(X|Không Muộn) = P(Muộn/Không Muộn)*P(Xấu/Không Muộn)*P(Mưa/Không Muộn)<br>
                    = 0 * ⅗ * ⅕  = 0.                    
                    </p>
                    """,unsafe_allow_html=True)
        
        st.markdown(f"""
                    <p> <strong>Step 2</strong>  We calculate P(X)<br> 
                    P(X) = P(X/Muộn) * P(Muộn) + P(X/Không Muộn) * P(Không Muộn)<br> 
                    = 6/125 * 5/10 + 0 * 5/10.<br>
                    = 3/125 = 0,024
                    </p>
                    """,unsafe_allow_html=True)
        
        st.markdown(f"""
                    <p> <strong>Step 3</strong>  We calculate P(y/X) of each feature:<br> 
                    P(Muộn/X) = (P(X/Muộn)*P(Muộn))/(P(X))<br>=(6/125*5/10)/(3/125)=1 (100%).<br> <br>
                    P(Không Muộn/X) = (P(X/(Không Muộn))*P(Không Muộn))/(P(X))<br>=(0*5/10)/(3/125)=0 (0%)<br>
                    </p>
                    """,unsafe_allow_html=True)
        st.markdown(f""" From the results, we compare the percentages of the 2 attributes and get the largest result<br>
                    So the So the probability of being Late is 100%.
                    """,unsafe_allow_html=True)
         
    with right: 
        image = Image.open('data/table_weather.png')
        st.image(image, caption='table example', width=600)
        
        
    st.write("## Advantages and Disadvantages of Naive Bayes Classifier")
    
    left, right = st.columns([2,2])
    
    with left: 
        st.write("### Advantages")
        st.markdown("""
                    * it is one of the fast and easy ML algorithms to predict a class of datasets
                    * It can be used for Binary as well as Multi-class Classifications.
                    * It performs well in Multi-class predictions as compared to the other Algorithms.
                    * It is the most popular choice for text classification problems
                    """)
    
    with right:
        st.write("### Disadvantages")
        st.write("Naive Bayes assumes that all features are independent or unrelated, so it cannot learn the relationship between features.")
        
    
    # Applications of Naïve Bayes Classifier
    st.write("### Applications of Naive Bayes Classifier")
    st.markdown("""
                * It is used for Credit Scoring.
                * It is used in medical data classification.
                * It can be used in real-time predictions because Naïve Bayes Classifier is an eager learner.
                * It is used in Text classification such as Spam filtering and Text classification.
                """)
    image = Image.open('data/spammail.jpg')
    _, center,_  = st.columns([1,2,1])
    with center:    
        st.image(image, caption='spam email example', width=600)
        
    
    st.markdown("---")
    
    st.write("## Types of Naive Bayes Model:")
    
    _, center, _ = st.columns([1,5,1])
    
    with center:
        image = Image.open('data/types.png') 
        st.image(image, caption='Types of Naive Bayes', width=800)    
    
    
    
    
    st.write("### Optimal Naive Bayes")
    left, right = st.columns([2,2])
    
    with left:
        st.markdown("""
                    * Optimal Naive Bayes selects the class that has the greatest posterior probability of happenings.
                    * As per the name, it is optimal. But it will go through all the possibilities, which is very slow and time-consuming.
                    """)
    with right: 
        image = Image.open('data/MPEE.png') 
        st.image(image, caption='MAP', width=500)    
    
    
    
    st.write("### Gaussian Naive Bayes")
    left, right = st.columns([2,2])
    with left:
        st.markdown("""
                * It is a s algorithm used when the features are continuous. 
                * The attributes present in the data should follow the rule of Gaussian distribution or normal distribution.
                * It remarkably quickens the search and greater than Optimal Naive Bayes                
                """)
    
    with right: 
        image = Image.open('data/NormalD.png') 
        st.image(image, caption='Example of a Normal Distribution', width=500)
        
    
    
    st.write("### Multinomial Naive Bayes")
    left, right = st.columns([2,2])
    with left:
        st.markdown("""
                * Multinomial Naive Bayes is used on documentation classification issues.
                * The features needed for this type are the frequency of the words converted from the document.
                """)
    
    with right: 
        image = Image.open('data/MNB.png') 
        st.image(image, caption='Equation', width=500)
    
    
    
    st.write("### Bernoulli Naive Bayes")
    left, right = st.columns([2,2])
    
    with left:
        st.markdown("""
                    * Bernoulli Naive Bayes is an algorithm that is useful for data that has binary or boolean attributes. 
                    * The attributes will have a value of yes or no, useful or not…""")
    
    with right: 
        image = Image.open('data/Beunoli.png') 
        st.image(image, caption='Equation', width=500)
    

    st.markdown("---")
    st.write("### Example")    
    st.write(" It's interesting to compare the performances of Gaussian and multinomial naive Bayes with the MNIST digit dataset")
    left, right = st.columns([3,2])
    
    with left:
        st.code("""
from sklearn.datasets import load_digits

from sklearn.model_selection import cross_val_score

digits = load_digits()

# Gaussian model 
gnb = GaussianNB()

# multinomial model 
mnb = MultinomialNB()

print("result of Gaussian model")
print(cross_val_score(gnb, digits.data, digits.target, scoring='accuracy', cv=10).mean())

print("result of multinomial model")                
print(cross_val_score(mnb, digits.data, digits.target, scoring='accuracy', cv=10).mean())

                """)
        
    with right: 
        code_button = st.button("Run code")
        
        if code_button:
            X, y = load_data_digits()
            with st.expander('data description'):
                st.dataframe(X)
            with st.expander('target description'):
                st.dataframe(y)
            
            st.write("Result of multinomial model".upper())
            gnb = GaussianNB()
            res = cross_val_score_model(gnb, X, y)
            st.write(res)
            
            st.write("result of Gaussian model".upper())
            mnb = MultinomialNB()
            res = cross_val_score_model(mnb, X, y)
            st.write(res)
    