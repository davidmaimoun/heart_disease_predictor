# By David Maimoun
# deployed the 13.03.23
import streamlit as st
import pandas as pd
import json
from streamlit_lottie import st_lottie
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


st.write("""
<style>
    h1 {
    color: rgb(255, 24, 49);
    }
    .result {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
</style>
""", unsafe_allow_html=True)

def load_lottiefile(filepath: str):
   with open(filepath, "r") as f:
      return json.load(f)

def returnLottie(path):
   return st_lottie(
      load_lottiefile(path),
      speed=1,
      reverse=False,
      loop=True,
      quality="high", # medium ; high
      # renderer="svg", # canvas
      height=None,
      width=None,
      key=None,
   )

st.markdown("<h1>Heart Disease Predictor</h1><br>",unsafe_allow_html=True)
        
bool = ('False', 'True')
sexes = ('Female', 'Male')
cp_angina = ("Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic")
thals = ("Normal", "Fixed defect", "Reversable defect")

st.sidebar.header("H.D.P App")
st.sidebar.markdown("<br>",unsafe_allow_html=True)

def user_input_features():
    age = st.sidebar.slider('1- Age', 0, 100, 25)
    
    sex = st.sidebar.selectbox('2- Sex',
                range(len(sexes)), format_func=lambda x: sexes[x])
   
    cp = st.sidebar.selectbox('3- Chest pain type', 
            range(len(cp_angina)), format_func=lambda x: cp_angina[x])
    
    trestbps = st.sidebar.slider('4- Resting Blood Pressure', 0, 200, 110)
    
    chol = st.sidebar.slider('5- Serum Cholesterol', 0, 600, 115)
    
    fbs = st.sidebar.selectbox('6- Fasting blood sugar',
            range(len(bool)), format_func=lambda x: bool[x])

    restecg = st.sidebar.selectbox('7- Resting electrocardiographic results: ', ["0", "1", "2"],
        help="""
        0: normal;
        1: having ST-T wave abnormality;
        2: showing probable or definite left ventricular hypertrophy.
        """)

    thalach = st.sidebar.slider('8- Max Heart Rate Achieved', 0, 220, 115)
    
    exang = st.sidebar.selectbox('9- Exercise Induced Angina',
            range(len(bool)), format_func=lambda x: bool[x])
    
    oldpeak = float(st.sidebar.slider('10- Oldpeak', 0.0, 10.0, 2.0, help='ST depression induced by exercise relative to rest'))
    
    slope = st.sidebar.selectbox("11- Slope",["0","1","2"], help='The slope of the peak exercise ST segment')
    
    ca = st.sidebar.selectbox("12- Number of major vessels",["0","1","2","3"])
    
    thal = float(st.sidebar.slider('13- Thalassemia',  0.0, 10.0, 3.0, help='3 = normal; 6 = fixed defect; 7 = reversable defect'))
    # thal = st.sidebar.selectbox('13- Thalassemia',
    #     range(len(thals)), format_func=lambda x: thals[x])    
    
    data = {
        'age': age, 
        'sex': sex, 
        'cp': cp+1, 
        'trestbps': trestbps, 
        'chol': chol, 
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach, 
        'exang': exang, 
        'oldpeak': oldpeak, 
        'slope': slope, 
        'ca': ca,
        'thal': thal
    }
    features = pd.DataFrame(data, index=[0])
    return features

my_data = user_input_features()

st.write("Patient data:")
st.dataframe(my_data)

df = pd.read_csv('heart.csv')
# df = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

X = df.drop(['target'], axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = LogisticRegression()
model.fit(X_train, y_train)

prediction = model.predict(my_data)
proba = model.predict_proba(my_data)

is_has_disease = "Has" if prediction[0] == 1 else "Doesn't Have"
style = "color:red" if prediction[0] == 1 else "color:dodgerblue"

col1, col2 = st.columns([1,3])
with col1:
    returnLottie("assets/doctor1.json")
with col2:
   st.markdown(f"""
    The patient 
    <span class='result' style={style}>{is_has_disease}</span> a Heart Disease, with a 
    probability of <span class='result' style={style}>{round(proba[0][1], 2)*100}%</span>.
""", unsafe_allow_html = True)

st.markdown('<br><br><p><i>By David Maimoun</p></i>',unsafe_allow_html=True)  
