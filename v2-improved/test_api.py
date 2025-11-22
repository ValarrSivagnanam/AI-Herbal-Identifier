import requests
import json
from pathlib import Path

# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
API_URL = "http://localhost:5000"

# -------------------------------------------
# TEST FUNCTIONS
# -------------------------------------------

def test_health_check():
    """Test if API is running"""
    print("\n" + "="*50)
    print("1. Testing Health Check")
    print("="*50)
    
    response = requests.get(f"{API_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_get_classes():
    """Test getting available classes"""
    print("\n" + "="*50)
    print("2. Testing Get Classes")
    print("="*50)
    
    response = requests.get(f"{API_URL}/classes")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_single_prediction(image_path):
    """Test single image prediction"""
    print("\n" + "="*50)
    print("3. Testing Single Image Prediction")
    print("="*50)
    print(f"Image: {image_path}")
    
    # Open and send image
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Predicted Class: {result['predicted_class']}")
        print(f"✓ Confidence: {result['confidence_percentage']}")
        print(f"\nAll Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name:15s}: {prob*100:5.2f}%")
    else:
        print(f"Error: {response.json()}")
    
    return response.status_code == 200

def test_batch_prediction(image_paths):
    """Test batch image prediction"""
    print("\n" + "="*50)
    print("4. Testing Batch Prediction")
    print("="*50)
    print(f"Number of images: {len(image_paths)}")
    
    files = []
    for img_path in image_paths:
        files.append(('images', open(img_path, 'rb')))
    
    response = requests.post(f"{API_URL}/predict_batch", files=files)
    
    for _, file in files:
        file.close()
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nProcessed {result['total_images']} images:")
        for pred in result['predictions']:
            if 'error' not in pred:
                print(f"\n  {pred['filename']}:")
                print(f"    → {pred['predicted_class']} ({pred['confidence_percentage']})")
            else:
                print(f"\n  {pred['filename']}: ERROR - {pred['error']}")
    else:
        print(f"Error: {response.json()}")
    
    return response.status_code == 200

def test_error_handling():
    """Test API error handling"""
    print("\n" + "="*50)
    print("5. Testing Error Handling")
    print("="*50)
    
    print("\nTest 5a: No image provided")
    response = requests.post(f"{API_URL}/predict")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    print("\nTest 5b: Empty filename")
    files = {'image': ('', b'')}
    response = requests.post(f"{API_URL}/predict", files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return True

# -------------------------------------------
# MAIN TEST RUNNER
# -------------------------------------------
def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*25)
    print("HERBAL PLANT CLASSIFICATION API - TEST SUITE")
    print("="*25)
    
    # Test 1: Health check
    test_health_check()
    
    # Test 2: Get classes
    test_get_classes()
    
    # Test 3: Single prediction
    test_image = "C:/Valarr/Herbal Plant Classification/dataset/test/brahmi/4853.jpg"
    
    if Path(test_image).exists():
        test_single_prediction(test_image)
    else:
        print(f"\n⚠ Warning: Test image not found at {test_image}")
        print("Please update the image path in the script")
    
    # Test 4: Batch prediction
    test_images = [
        "C:/Valarr/Herbal Plant Classification/dataset/test/aloevera/4353.jpg",
        "C:/Valarr/Herbal Plant Classification/dataset/test/brahmi/4857.jpg",
        "C:/Valarr/Herbal Plant Classification/dataset/test/tulasi/3646.jpg"
    ]
    
    existing_images = [img for img in test_images if Path(img).exists()]
    if existing_images:
        test_batch_prediction(existing_images)
    else:
        print("\n Warning: No test images found for batch prediction")
    
    # Test 5: Error handling
    test_error_handling()
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50 + "\n")

# -------------------------------------------
# STANDALONE PREDICTION FUNCTION
# -------------------------------------------
def predict_image(image_path):
    """
    Convenient function to predict a single image
    
    Usage:
        result = predict_image("path/to/image.jpg")
        print(result['predicted_class'])
    """
    with open(image_path, 'rb') as img_file:
        files = {'image': img_file}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Prediction failed: {response.json()}")

# -------------------------------------------
# RUN TESTS
# -------------------------------------------
if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("\n ERROR: Could not connect to API")
        print("Make sure the Flask server is running on http://localhost:5000")
        print("\nTo start the server, run:")
        print("  python app.py")
    except Exception as e:
        print(f"\n ERROR: {e}")