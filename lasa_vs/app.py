import os
import speech_recognition as sr
from pydub import AudioSegment
import nltk
from nltk.corpus import cmudict
from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

nltk.download('cmudict', quiet=True)
cmu_dict = cmudict.dict()


LASA_MEDICINES = {
    "zantac": ["Z532", "zantac 150mg", "zantac tablet 150mg"],
    "soma": ["S530", "soma tablet 350mg"],
    "criminal": ["C654", "criminal tablet 12mg"],
    "hydroxyzine": ["H362", "hydroxyzine"],
    "hydralazine": ["H364", "hydralazine"],
    "celexa": ["C420", "celexa"],
    "celebrex": ["C416", "celebrex"],
    "clonidine": ["C435", "clonidine"],
    "clonazepam": ["C435", "clonazepam"],
    "zyrtec": ["Z632", "zyrtec"],
    "lamictal": ["L523", "lamictal"],
    "lamisil": ["L524", "lamisil"],
    "trazodone": ["T621", "trazodone"],
    "tramadol": ["T622", "tramadol"],
    "metformin": ["M431", "metformin", "glycomet", "glycomet gp"],
    "metronidazole": ["M432", "metronidazole"],
    "norvasc": ["N510", "norvasc"],
    "navane": ["N511", "navane"],
    "zyprexa": ["Z634", "zyprexa"],
    "zyvox": ["Z635", "zyvox"],
    "ritalin": ["R456", "ritalin"],
    "ribavirin": ["R457", "ribavirin"],
    "prednisone": ["P342", "prednisone"],
    "prednisolone": ["P343", "prednisolone"],
    "dopamine": ["D212", "dopamine"],
    "dobutamine": ["D213", "dobutamine"],
    

  
    "telmisartan": ["T431", "telmisartan"],
    "pan-d": ["P561", "pantoprazole domperidone", "pan d", "pan-d"],
    "pantoprazole": ["P562", "pantoprazole"],
    "amlodipine": ["A333", "amlodipine"],
    "atenolol": ["A334", "atenolol"],
    "atorvastatin": ["A336", "atorvastatin"],
    "augmentin": ["A339", "augmentin", "amoxicillin clavulanic"],
    "ryzodeg": ["R888", "ryzodeg", "insulin degludec", "insulin aspart"],
    "udiliv": ["U777", "ursodeoxycholic acid", "udiliv"],
    "becosules": ["B666", "becosules", "multivitamin"],
    "ibuprofen": ["I232", "ibuprofen"],
    "diclofenac": ["D233", "diclofenac"],
    "levocetirizine": ["L234", "levocetirizine"],
    "montelukast": ["M235", "montelukast"]
}

def get_phonemes(word):
    """Retrieve phonemes for a given word from CMU dictionary"""
    word = word.lower().strip()
    
    for suffix in [' tablet', ' mg', ' capsule']:
        word = word.replace(suffix, '')
    
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

def similarity_score(word1, word2):
   
    string_similarity = SequenceMatcher(None, word1.lower(), word2.lower()).ratio() * 100
    
    levenshtein_similarity = calculate_levenshtein_similarity(word1, word2)
    
    phonemes1 = get_phonemes(word1)
    phonemes2 = get_phonemes(word2)
    
    if phonemes1 and phonemes2:
        code1 = soundex(phonemes1)
        code2 = soundex(phonemes2)
        phonetic_similarity = SequenceMatcher(None, code1, code2).ratio() * 100
        
        return (string_similarity * 0.3 + levenshtein_similarity * 0.3 + phonetic_similarity * 0.4)
    

    return (string_similarity * 0.5 + levenshtein_similarity * 0.5)

def check_LASA(input_text):
    """Check if input text contains any LASA medications"""
    input_words = input_text.lower().split()
    results = []
    
    
    for word in input_words:
        word = word.strip()
        
        base_word = word
        for suffix in [' tablet', ' mg', ' capsule']:
            base_word = base_word.replace(suffix, '')
        
        
        for med_base, variations in LASA_MEDICINES.items():
            
            base_similarity = similarity_score(base_word, med_base)
            
        
            for variation in variations:
                var_similarity = similarity_score(word, variation)
                similarity = max(base_similarity, var_similarity)
                
                if similarity >= 50:  
                    
                    similar_meds = []
                    for other_med, other_variations in LASA_MEDICINES.items():
                        if other_med != med_base:
                            other_similarity = similarity_score(med_base, other_med)
                            if other_similarity >= 50:
                        
                                lev_distance = levenshtein_distance(med_base, other_med)
                                similar_meds.append({
                                    "name": other_med,
                                    "similarity": round(other_similarity, 1),
                                    "levenshtein_distance": lev_distance
                                })
                    
                    similar_meds.sort(key=lambda x: x['similarity'], reverse=True)
                    
                    results.append({
                        "input_word": word,
                        "matched_medicine": med_base,
                        "similarity": round(similarity, 1),
                        "similar_medicines": similar_meds
                    })
                    break  
    
    return results

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
