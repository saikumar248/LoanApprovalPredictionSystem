from flask import Flask, render_template, request, redirect, url_for,session,render_template_string
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.use('Agg')
from io import BytesIO
import pickle
from sklearn.metrics import plot_confusion_matrix

from loaneligibility import values

X_test, y_test = values()

app = Flask(__name__)
app.secret_key = os.urandom(24)

import sqlite3

connection = sqlite3.connect('validation.db',check_same_thread=False)
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS
users(user_id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, password TEXT)"""

cursor.execute(command)

# cursor.execute("INSERT INTO users  VALUES (1, 'Sai', 'email@gmail','Sai733kumar@')")
connection.commit()


@app.route('/login_validation',methods=['POST'])
def login_validation():

    email=request.form.get('email')
    password=request.form.get('password')

    cursor.execute("""SELECT *From `users` WHERE `email` LIKE '{}' AND `password` LIKE '{}' """
                   .format(email,password))
    users = cursor.fetchall()
    if len(users)>0:
        session['user_id'] = users[0][0]
        return redirect('/main')
    else:
        return redirect('/')

@app.route('/add_user',methods=['POST'])
def add_user():
    name=request.form.get('uname')
    email=request.form.get('uemail')
    password=request.form.get('upassword')

    cursor.execute(""" Insert into `users` (`user_id`,`name`, `email`, `password`) values 
    (NULL,'{}','{}','{}')""".format(name, email, password))
    connection.commit()
    cursor.execute(""" SELECT *FROM `users` WHERE `email` LIKE '{}'""".format(email))
    myuser=cursor.fetchall()
    session['user_id']=myuser[0][0]
    return redirect('/main')

@app.route('/')
def home():
        return render_template('home.html', )
   


model = pickle.load(open('loan_eligibility_adaboost.pkl', 'rb'))


@app.route('/main')
def main():
    if 'user_id' in session:
        return render_template('main.html', )
    else:
        return redirect('/')


@app.route("/predict", methods=['GET','POST'])
def predict():

    if request.method == 'POST':
        
        
        ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Dependents',
       'Loan_Amount_Term', 'Credit_History', 
       
       'Male', 'Married_Yes',
       'Self_Employed_Yes', 'Not_Graduate', 'Semiurban', 'Urban']
        
        ApplicantIncome = float(request.form['ApplicantIncome'])
        
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        
        LoanAmount = float(request.form['LoanAmount'])
        
        Dependents = float(request.form['Dependents'])
        
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        
        Credit_History = float(request.form['Credit_History'])
        
        
        
        Gender = request.form['Gender']
        if (Gender == 'Male'):
            Male = 1
            
        else :
            Male = 0
            
            
            
        Married = request.form['Married']
        if (Married == 'Yes'):
            Married_Yes = 1
            
        else :
            Married_Yes = 0
            
            
            
        Self_Employed = request.form['Self_Employed']
        if (Self_Employed == 'Yes'):
            Self_Employed_Yes = 1
            
        else :
            Self_Employed_Yes = 0
            
            
            
        Education = request.form['Education']
        if (Education == 'No'):
            Not_Graduate = 1
            
        else :
            Not_Graduate = 0
            
            
            
        Property_Area = request.form['Property_Area']                
        if (Property_Area == 'Urban'):
            Urban = 1
            Semiurban = 0
            
            
        elif (Property_Area == 'Semiurban'):
            Urban = 0
            Semiurban = 1
            
        else:
            Urban = 0
            Semiurban = 0
            
        prediction=model.predict([[ApplicantIncome, CoapplicantIncome, LoanAmount, Dependents, Loan_Amount_Term, Credit_History, Male, Married_Yes, Self_Employed_Yes, Not_Graduate, Semiurban, Urban]])
        
        output = prediction

        if output == 0:
            return render_template('predict.html',prediction_text="The applicant is Not Eligible for Loan")
        elif output == 1:
            return render_template('predict.html',prediction_text="The applicant is Eligible for Loan")
    else:
        return render_template('predict.html')



@app.route('/temp')
def temp():
    if 'user_id' in session:
        data = pd.read_csv('loan.csv')
        html_table=data.to_html()
        # with open('templates/temp.html', 'w') as f:
        #     f.write(html_table)

        # print(data.head())
        return render_template('temp.html',html_table=html_table)
    else:
        return redirect('/')

@app.route('/chart')
def chart():
    if 'user_id' in session:
        df = pd.read_csv('loan.csv')
        result_counts = df['Loan_Status'].value_counts()


#        plt.bar(result_counts.index, result_counts.values)
#        plt.xlabel("Result")
#        plt.ylabel("Count")
#        plt.title("Count of Yes and No Results")
#        plt.show()

        plt.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title("Distribution of Yes and No Results")

    # plt.show()
        image_path = os.path.join('static', 'chart.png')
    
        if os.path.exists(image_path):
            os.remove(image_path)
        plt.savefig(image_path)

        plt.close()
        plot_confusion_matrix(model, X_test, y_test)
        plt.title('Confusion Matrix\n')
        plt.show()
        imagepath = os.path.join('static', 'confuse.png')
    
        if os.path.exists(imagepath):
            os.remove(imagepath)
        plt.savefig(imagepath)

        plt.close()
    # html_chart = '<img src="{{ url_for(\'static\', filename=\'chart.png\') }}" alt="Line Chart">'
        return render_template('chart.html')
    else:
        return redirect('/')
    # image_stream = BytesIO()
    # plt.savefig(image_stream, format='png')
   

    # image_data= image_stream.getvalue()
    # return render_template_string('<img src="data:image/png;base64,{{ image_data }}" alt="Line Chart">', image_data=image_data)





@app.route('/signin')
def signin():
    return render_template('signin.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

@app.route('/signout')
def signout():
    if 'user_id' not in session:
        return redirect('/')
    else:
        session.pop('user_id')
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
