from flask import Flask, render_template, request

app = Flask(__name__)
@app.route('/')
def home():

    return render_template('index.html')

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        import numpy as np
        from joblib import load
        new_loanapp = load('file.joblib')
        refno = request.form.to_dict() #.to_dict() method is used to convert a dataframe into a dictionary of series or list
        print('refno = request.form.to_dict()')
        print(refno)
        refno = list(refno.values())  # returns a list of all the values available in a given dictionary
        print('refno = list(refno.values())')
        print(refno)
        refno = list(map(int, refno))
        print('refno = list(map(int, refno)')
        print(refno)
        refno = np.array(refno).reshape(1, 11)
        print('refno = np.array(refno).reshape(1, 11)')
        print(refno)
        result = new_loanapp.predict(refno)
        if int(result)== 1:
            prediction ='Congratulations! Your Loan is Approved'
        else:
            prediction ='Sorry! Your Loan Application Is REJECTED'
        return render_template("result.html", prediction = prediction)

if __name__ == "__main__":

    app.run(debug=True)