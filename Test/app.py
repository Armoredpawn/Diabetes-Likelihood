from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the saved model
filename = r"saved_model.sav"
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def submit():
    # Default values for the form
    default_values = {
        'BP': 0,
        'HC': 0,
        'CC': 0,
        'BMI': 30,
        'SM': 0,
        'ST': 0,
        'HD': 0,
        'EX': 1,
        'FR': 1,
        'VE': 1,
        'DR': 0,
        'CO': 1,
        'ND': 0,
        'RH': 1,
        'MH': 0,
        'PH': 0,
        'DW': 0,
        'level0': '',
        'level': '',
        'level2': '',
        'level3': ''
    }
    
    # Initialize form_data to default values
    form_data = default_values.copy()

    if request.method == 'POST':
        # Collect the form data from the submitted form
        form_data.update({
            'BP': request.form.get('BP', default_values['BP']),
            'HC': request.form.get('HC', default_values['HC']),
            'CC': request.form.get('CC', default_values['CC']),
            'BMI': request.form.get('BMI', default_values['BMI']),
            'SM': request.form.get('SM', default_values['SM']),
            'ST': request.form.get('ST', default_values['ST']),
            'HD': request.form.get('HD', default_values['HD']),
            'EX': request.form.get('EX', default_values['EX']),
            'FR': request.form.get('FR', default_values['FR']),
            'VE': request.form.get('VE', default_values['VE']),
            'DR': request.form.get('DR', default_values['DR']),
            'CO': request.form.get('CO', default_values['CO']),
            'ND': request.form.get('ND', default_values['ND']),
            'RH': request.form.get('RH', default_values['RH']),
            'MH': request.form.get('MH', default_values['MH']),
            'PH': request.form.get('PH', default_values['PH']),
            'DW': request.form.get('DW', default_values['DW']),
            'level0': request.form.get('level0', default_values['level0']),
            'level': request.form.get('level', default_values['level']),
            'level2': request.form.get('level2', default_values['level2']),
            'level3': request.form.get('level3', default_values['level3']),
        })

        # Convert form values to numeric types where applicable
        for key, value in form_data.items():
            if isinstance(value, str):  # Only apply .upper() on strings
                if value.upper() == "YES":
                    form_data[key] = 1
                elif value.upper() == "NO":
                    form_data[key] = 0
                elif value.upper() == "MALE":
                    form_data[key] = 1
                elif value.upper() == "FEMALE":
                    form_data[key] = 0
                else:
                    try:
                        # Convert to int or float if possible
                        if '.' in value:
                            form_data[key] = float(value)  # Convert to float
                        else:
                            form_data[key] = int(value)  # Convert to int
                    except ValueError:
                        form_data[key] = 0  # Set to 0 if conversion fails
            else:
                try:
                    # For non-string values (integers or floats), just convert directly if necessary
                    form_data[key] = float(value) if '.' in str(value) else int(value)
                except ValueError:
                    form_data[key] = 0  # Default value if conversion fails
        
        # Now process the input_values
        input_values = list(form_data.values())
        array = np.array(input_values)
        X_test = array.reshape((1, -1))
        probs = model.predict_proba(X_test)
        
        s2 = ""
        space = False
        for i in probs:
            i = str(i)
            for j in i:
                if j == " ":
                    space = True
                elif j == "[" or j == "]":
                    continue
                else:
                    if space == True:
                        s2 += j
        s2 = float(s2) * 100
        if s2 < 12.079556854343798:
        	s2 = "LOW"
        else:
        	s2 = "HIGH"
        s2 = "Risk of developing diabetes: " + s2

        # Render the page with the result and the form data (including user inputs)
        return render_template('index.html', input_values=s2, form_data=form_data)

    # Render the page with default values for GET requests
    return render_template('index.html', form_data=form_data)

if __name__ == "__main__":
    app.run(debug=True)
