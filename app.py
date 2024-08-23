from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from inference_sdk import InferenceHTTPClient, InferenceConfiguration

app = Flask(__name__)
#app.config['SECRET_KEY'] = 'your_secret_key'  # Replace with a strong secret key
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded images

# Initialize InferenceHTTPClient
#custom_configuration = InferenceConfiguration(confidence_threshold=0.0)
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="egr6yCdf3VYRLzWtMq4Y"
)
#client.configure(custom_configuration)

def run_prediction(image_path):
    try:
        # Run prediction using InferenceHTTPClient
        result = client.run_workflow(
            workspace_name="nishant-raghuwanshi-deep-learning",
            workflow_id="custom-workflow",
            images={"image": image_path}
        )
        return result
    except Exception as e:
        return {'error': str(e)}

def extract_and_format(result,confidence_threshold):
    formatted_predictions = []
    
    for entry in result:
        # Check if 'predictions' key is present in entry
        if 'predictions' in entry:
            predictions = entry['predictions']
            # Check if 'image' key is present in predictions
            if 'image' in predictions:
                image_info = predictions['image']
                print(f"Image Dimensions: {image_info['width']}x{image_info['height']}")
            
            # Check if 'predictions' key is present in predictions
            if 'predictions' in predictions:
                for pred in predictions['predictions']:
                    if pred.get('confidence', 0) >= confidence_threshold:
    
                        print("\nPrediction:")
                        print(f"  - Width: {pred.get('width', 0)}")
                        print(f"  - Height: {pred.get('height', 0)}")
                        print(f"  - X: {pred.get('x', 0)}")
                        print(f"  - Y: {pred.get('y', 0)}")
                        print(f"  - Confidence: {pred.get('confidence', 0):.2f}")
                        print(f"  - Class ID: {pred.get('class_id', 'N/A')}")
                        print(f"  - Class: {pred.get('class', 'Unknown')}")
                        print(f"  - Detection ID: {pred.get('detection_id', 'N/A')}")
                        print(f"  - Parent ID: {pred.get('parent_id', 'N/A')}")
                        print("  - Points:")
                        for point in pred.get('points', []):
                            print(f"    - X: {point.get('x', 0)}, Y: {point.get('y', 0)}")
                        
                        # Append formatted prediction to the list
                        formatted_predictions.append({
                            'class': pred.get('class', 'Unknown'),
                            'x': pred.get('x', 0),
                            'y': pred.get('y', 0),
                            'width': pred.get('width', 0),
                            'height': pred.get('height', 0),
                            'confidence': pred.get('confidence', 0),
                            'class_id': pred.get('class_id', 'N/A'),
                            'detection_id': pred.get('detection_id', 'N/A'),
                            'parent_id': pred.get('parent_id', 'N/A'),
                            'points': pred.get('points', [])
                        })
    
    return formatted_predictions

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file.filename != '':
            confidence = float(request.form.get('confidence', 50)) / 100.0  # Convert slider value to a decimal
            print(confidence)
            custom_configuration = InferenceConfiguration(confidence_threshold=confidence)
            client.configure(custom_configuration)

            filename = secure_filename(uploaded_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(image_path)

            prediction_result = run_prediction(image_path)
            if 'error' in prediction_result:
                return render_template('index.html', error_message=prediction_result['error'])
            else:
                predictions = extract_and_format(prediction_result,confidence)
                return render_template('index.html', predictions=predictions)
        else:
            return render_template('index.html', error_message='No image selected for upload')
    return render_template('index.html')

if __name__ == '__main__':
    import os
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
