import cv2
import numpy as np
from matplotlib import pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def calculate_statistics(image):
    mean = np.mean(image)
    variance = np.var(image)
    stddev = np.std(image)
    return mean, variance, stddev

def plot_histogram(image, title):
    plt.figure(figsize=(10, 5))
    if len(image.shape) == 2:  # Grayscale image
        plt.hist(image.ravel(), bins=256, range=[0, 256], color='black')
    else:  # Color image
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], f'{title}.png'))
    plt.close()

def normalize_image(image):
    normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized_image

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            
            # Convert to grayscale if image is colored
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate statistics
            mean, variance, stddev = calculate_statistics(gray_image)
            stats = f'Mean: {mean:.2f}, Variance: {variance:.2f}, Standard Deviation: {stddev:.2f}'
            
            # Plot and save histograms
            plot_histogram(gray_image, 'grayscale_hist')
            normalized_image = normalize_image(gray_image)
            plot_histogram(normalized_image, 'normalized_hist')
            
            return render_template('results.html', stats=stats, 
                                original_image=os.path.join('uploads', file.filename), 
                                grayscale_hist='uploads/grayscale_hist.png', 
                                normalized_hist='uploads/normalized_hist.png')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
