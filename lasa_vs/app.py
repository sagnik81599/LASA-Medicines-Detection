import os
import string
import json
import speech_recognition as sr
from pydub import AudioSegment
import nltk
from nltk.corpus import cmudict
from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance
from flask import Flask, request, jsonify, render_template
import cloudinary
import cloudinary.uploader
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import shutil
import google.generativeai as genai


app = Flask(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Configure tesseract path
tesseract_path = shutil.which('tesseract')
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # Fallback to common locations
    common_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\ProgramData\chocolatey\bin\tesseract.exe',
        r'C:\Users\ASUS\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
        r'C:\Windows\System32\tesseract.exe'
    ]
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUD_NAME'),
    api_key=os.getenv('API_KEY'),
    api_secret=os.getenv('API_SECRET'),
    secure=True
)

nltk.download('cmudict', quiet=True)
cmu_dict = cmudict.dict()

with open('medicines.json', 'r') as f:
    LASA_MEDICINES = json.load(f)

def clean_word(word):
    """Remove punctuation and extra whitespace from word"""
    word = word.lower().strip()
    word = word.translate(str.maketrans('', '', string.punctuation))
    return word.strip()

def get_phonemes(word):
    """Retrieve phonemes for a given word from CMU dictionary"""
    word = clean_word(word)
    
    for suffix in [' tablet', ' mg', ' capsule']:
        word = word.replace(suffix, '').strip()
    
    if word in cmu_dict:
        return cmu_dict[word][0]
    return None

def soundex(phonemes):
  
    if not phonemes:
        return None
    
    
    soundex_dict = {
        'BFPV': '1',
        'CGJKQSXZ': '2',
        'DT': '3',
        'L': '4',
        'MN': '5',
        'R': '6',
        'AEIOUHWY': '0'  
    }

    code = phonemes[0][0].upper()
    
    for phoneme in phonemes[1:]:
        for key, value in soundex_dict.items():
            if any(letter in phoneme.upper() for letter in key):
                if len(code) == 1 or value != code[-1]:
                    if value != '0':  
                        code += value
                break
    
    return code[:4].ljust(4, '0')

def calculate_levenshtein_similarity(word1, word2):
 
    max_length = max(len(word1), len(word2))
    if max_length == 0:
        return 100.0
    distance = levenshtein_distance(word1.lower(), word2.lower())
    return (1 - distance / max_length) * 100

def calculate_phonetic_similarity(phonemes1, phonemes2):
    """Calculate similarity between two phoneme sequences"""
    if not phonemes1 or not phonemes2:
        return 0
    
    phoneme_str1 = ''.join(phonemes1).lower()
    phoneme_str2 = ''.join(phonemes2).lower()
    
    max_len = max(len(phoneme_str1), len(phoneme_str2))
    if max_len == 0:
        return 100
    
    distance = levenshtein_distance(phoneme_str1, phoneme_str2)
    return max(0, (1 - distance / max_len) * 100)

def similarity_score(word1, word2):
   
    string_similarity = SequenceMatcher(None, word1.lower(), word2.lower()).ratio() * 100
    
    levenshtein_similarity = calculate_levenshtein_similarity(word1, word2)
    
    phonemes1 = get_phonemes(word1)
    phonemes2 = get_phonemes(word2)
    
    if phonemes1 and phonemes2:
        phonetic_similarity = calculate_phonetic_similarity(phonemes1, phonemes2)
        code1 = soundex(phonemes1)
        code2 = soundex(phonemes2)
        soundex_similarity = SequenceMatcher(None, code1, code2).ratio() * 100
        
        combined_phonetic = (phonetic_similarity * 0.6 + soundex_similarity * 0.4)
        return (string_similarity * 0.15 + levenshtein_similarity * 0.15 + combined_phonetic * 0.7)
    

    return (string_similarity * 0.4 + levenshtein_similarity * 0.6)

def check_LASA(input_text):
    """Check if input text contains any LASA medications"""
    input_words = input_text.lower().split()
    results_dict = {}
    
    for word in input_words:
        word = clean_word(word)
        if not word:
            continue
        
        base_word = word
        for suffix in [' tablet', ' mg', ' capsule']:
            base_word = base_word.replace(suffix, '')
        
        for med_base, med_data in LASA_MEDICINES.items():
            base_similarity = similarity_score(base_word, med_base)
            
            for variation in med_data['aliases']:
                var_similarity = similarity_score(word, variation)
                similarity = max(base_similarity, var_similarity)
                
                if similarity >= 50:
                    if med_base not in results_dict or similarity > results_dict[med_base]['similarity']:
                        results_dict[med_base] = {
                            "input_word": word,
                            "matched_medicine": med_base,
                            "code": med_data['code'],
                            "purpose": med_data['purpose'],
                            "similarity": round(similarity, 1),
                            "similar_medicines": []
                        }
                    break
    
    results = list(results_dict.values())
    
    for result in results:
        similar_meds = []
        med_base = result['matched_medicine']
        for other_med, other_med_data in LASA_MEDICINES.items():
            if other_med != med_base:
                other_similarity = similarity_score(med_base, other_med)
                if other_similarity >= 50:
                    lev_distance = levenshtein_distance(med_base, other_med)
                    similar_meds.append({
                        "name": other_med,
                        "code": other_med_data['code'],
                        "purpose": other_med_data['purpose'],
                        "similarity": round(other_similarity, 1),
                        "levenshtein_distance": lev_distance
                    })
        
        similar_meds.sort(key=lambda x: x['similarity'], reverse=True)
        result['similar_medicines'] = similar_meds
    
    return results

def extract_text_from_image(image_path):
    """Extract text from prescription image using OCR"""
    try:
        # Open the image
        image = Image.open(image_path)

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Extract text using pytesseract
        text = pytesseract.image_to_string(image)

        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def extract_medications_with_gemini(text):
    """Extract medication names from text using Gemini AI"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Extract only the medication names from the following prescription text. List them separated by commas, nothing else. If no medications found, return empty string:\n\n{text}"
        response = model.generate_content(prompt)
        medications = response.text.strip()
        return medications
    except Exception as e:
        return f"Error extracting medications: {str(e)}"

@app.route('/')
def index():
   
    return render_template('index.html')

@app.route('/check_lasa', methods=['POST'])
def check_lasa_route():
   
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400

        results = check_LASA(text)
        
        if not results:
            return jsonify({
                "message": "No LASA medications found with significant similarity.",
                "similar_medicines": []
            })
        
        return jsonify({
            "message": "LASA medications found!",
            "similar_medicines": results
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Handle audio file upload"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        
        results = process_audio(file_path)
        
        
        try:
            os.remove(file_path)
        except:
            pass  
            
        return jsonify(results)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle prescription image upload and extract medicine names"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(file, folder="prescriptions")

            # Get the secure URL
            image_url = upload_result['secure_url']

            # For OCR processing, we need to download the image temporarily
            # Since Cloudinary provides the URL, we can use it directly with pytesseract
            # But for simplicity, let's save the file temporarily and process it
            temp_path = os.path.join('uploads', 'temp_' + file.filename)
            file.seek(0)  # Reset file pointer
            file.save(temp_path)

            # Extract text from image
            extracted_text = extract_text_from_image(temp_path)

            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass

            if extracted_text.startswith("Error"):
                return jsonify({"error": extracted_text}), 500

            # Extract medication names using Gemini
            filtered_medications = extract_medications_with_gemini(extracted_text)

            if filtered_medications.startswith("Error"):
                return jsonify({"error": filtered_medications}), 500

            # Check for LASA medications in filtered medications
            results = check_LASA(filtered_medications)

            return jsonify({
                "message": "Image processed successfully",
                "filtered_medications": filtered_medications,
                "image_url": image_url,
                "similar_medicines": results
            })

        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500

def process_audio(file_path):
    """Process audio file and detect LASA errors"""
    try:
       
        if file_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(file_path)
            wav_path = file_path.rsplit('.', 1)[0] + '.wav'
            audio.export(wav_path, format="wav")
        else:
            wav_path = file_path

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)

            try:
                text = recognizer.recognize_google(audio_data)
                results = check_LASA(text)
                return {"similar_medicines": results}
            except sr.UnknownValueError:
                return {"error": "Could not understand audio."}
            except sr.RequestError as e:
                return {"error": f"Speech recognition service error: {e}"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
  
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
