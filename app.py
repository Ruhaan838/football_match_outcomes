from flask import Flask, render_template_string
import pandas as pd
import os

app = Flask(__name__)

# 📌 Hardcoded path to your CSV file (edit this as needed)
CSV_PATH = 'data/football_data.csv'  # <-- update this!

HTML_TEMPLATE = '''
<!doctype html>
<title>CSV Viewer</title>
<h1>CSV Viewer - Predefined Path</h1>
{% if error %}
    <p style="color: red;">{{ error }}</p>
{% endif %}
{% if table %}
    {{ table | safe }}
{% endif %}
'''

@app.route('/')
def view_csv():
    error = None
    table_html = None

    if not os.path.isfile(CSV_PATH):
        error = f"File not found: {CSV_PATH}"
    elif not CSV_PATH.endswith('.csv'):
        error = "Hardcoded path is not a CSV file."
    else:
        try:
            df = pd.read_csv(CSV_PATH)
            table_html = df.to_html(classes='table table-bordered', index=False)
        except Exception as e:
            error = f"Error reading file: {str(e)}"

    return render_template_string(HTML_TEMPLATE, table=table_html, error=error)

if __name__ == '__main__':
    app.run(debug=True)
