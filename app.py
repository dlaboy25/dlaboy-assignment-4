# app.py

import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for
from lsa import LSAModel
import os

app = Flask(__name__)

# Initialize the LSA model
lsa_model = LSAModel(n_components=100)
lsa_model.load_data()
lsa_model.fit()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        return redirect(url_for('results', query=query))
    return render_template('index.html')

@app.route('/results')
def results():
    query = request.args.get('query')
    if not query:
        return redirect(url_for('index'))
    top_docs = lsa_model.process_query(query)
    # Prepare data for visualization
    scores = [score for _, score in top_docs]
    doc_labels = ['Doc {}'.format(i+1) for i in range(len(scores))]
    # Generate bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(doc_labels, scores, color='#3f51b5')
    ax.set_xlabel('Documents')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Top 5 Documents Similarity Scores')
    # Save the plot to a static file
    plot_path = os.path.join('static', 'plot.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    # Render results template
    return render_template('results.html', query=query, top_docs=top_docs)

if __name__ == '__main__':
    app.run(debug=True, port=3000)
