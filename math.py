import streamlit as st
from PIL import Image

im = Image.open("favicon.ico")
st.set_page_config(page_title="Naive app",page_icon=im,layout='wide')


left, center, _ = st.columns([1, 2, 1])

st.markdown("""
<style>
.big-font {
    font-size:300 !important;
}
</style>
""", unsafe_allow_html=True)



with center: 
    st.subheader("Example: ")
    image_example = Image.open('data/example.png')
    st.image(image_example, caption='Source code', width=1000)



st.title("Applications of Naive Bayes algorithm")
st.markdown("""
            * Spam filtering: bl a 
            * Text classification
            * Sentiment analysis
            * Recommender systems
            """)


# with left:
st.markdown("#### How does it work?")
# st.markdown("""Naive Bayes is an algorithm which is commonly used in natural language processing (NLP) tasks such as spam filtering, sentiment analysis, classification, recommendation. 
#             It is based on Bayes' Theorem as shown below.""")
text = """Naive Bayes is an algorithm which is commonly used in natural language processing (NLP) tasks such as spam filtering, sentiment analysis, classification, recommendation. 
It is based on Bayes' Theorem as shown below."""
st.text(text)

st.latex(r'''P(A|B) = \frac{P(B|A)P(A)}{P(B)}''')
# st.latex(r'''
#     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
#     \sum_{k=0}^{n-1} ar^k =
#     a \left(\frac{1-r^{n}}{1-r}\right)
#     ''')
st.markdown("""
            * P(A|B) = probability of A if B occurs
            * P(B|A) = probability of B if A occurs
            * P(A) = probability of A
            * P(B) = probability of A
            """)


st.title("In scikit-learn, there are 3 types naive bayes algorithms.")
st.subheader("Gaussian Naive Bayes algorithm")
st.write("""When we have continuous attribute values, we made an assumption that the values associated with each class are distributed according to Gaussian or Normal distribution.""" )
st.write("The probability distribution of xi given a class can be computed by the following equation")
image = Image.open('gaussian.png')
st.image(image, caption='Gaussian Naive Bayes algorithm')


# Multinomial Naïve Bayes algorithm
st.subheader("Multinomial Naïve Bayes algorithm")
st.write("Multinomial Naive Bayes algorithm is a probabilistic learning method that is mostly used in Natural Language Processing (NLP)")
st.write("The algorithm is based on the Bayes theorem and predicts the tag of a text such as a piece of email or newspaper article")



st.subheader("Bernoulli Naive Bayes algorithm")
