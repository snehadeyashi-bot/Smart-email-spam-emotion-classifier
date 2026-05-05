from flask import Flask, render_template, request
import joblib
import re
import string
from textblob import TextBlob

app = Flask(__name__)

# Load the trained model safely
try:
    spam_model = joblib.load('spam_model.pkl')
except Exception as e:
    spam_model = None
    print(f"Error loading model: {e}")

def preprocess_text(text):
    """
    Preprocess the text: lowercase, remove URLs, remove punctuation.
    """
    # Lowercase conversion
    text = text.lower()
    # URL cleaning
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Punctuation removal
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def analyze_emotion_and_tone(text):
    """
    Detailed emotional intelligence analysis.
    Returns a dictionary of insights.
    """
    urgent_keywords = ['asap', 'urgent', 'immediately', 'emergency', 'hurry', 'quick', 'deadline', 'required', 'important']
    angry_keywords = ['hate', 'frustrated', 'disappointed', 'angry', 'mad', 'terrible', 'worst', 'unacceptable', 'bad']
    happy_keywords = ['great', 'awesome', 'excellent', 'happy', 'thrilled', 'love', 'fantastic', 'wonderful', 'glad', 'thanks']
    professional_keywords = ['sincerely', 'regards', 'inform', 'attached', 'meeting', 'schedule', 'review', 'forward', 'dear', 'team']
    
    words = text.lower().split()
    
    urgency_count = sum(1 for w in words if w in urgent_keywords)
    angry_count = sum(1 for w in words if w in angry_keywords)
    happy_count = sum(1 for w in words if w in happy_keywords)
    prof_count = sum(1 for w in words if w in professional_keywords)
    
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Determine Primary Emotion
    primary_emotion = 'Neutral'
    if urgency_count > 0 and urgency_count >= angry_count:
        primary_emotion = 'Urgent'
    elif angry_count > 0 or polarity < -0.3:
        primary_emotion = 'Angry'
    elif happy_count > 0 or polarity > 0.4:
        primary_emotion = 'Happy'
    elif prof_count >= 1:
        primary_emotion = 'Professional'
        
    # Calculate Urgency Meter
    urgency_score = min(100, urgency_count * 25 + (50 if 'urgent' in words or 'asap' in words else 0))
    if primary_emotion == 'Urgent': urgency_score = max(75, urgency_score)
    
    # Tone Breakdown
    total_signals = urgency_count + angry_count + happy_count + prof_count + 1 
    breakdown = {
        'Urgent': round((urgency_count / total_signals) * 100),
        'Angry': round((angry_count / total_signals) * 100),
        'Happy': round((happy_count / total_signals) * 100),
        'Professional': round((prof_count / total_signals) * 100),
    }
    
    # Adjust for polarity
    if polarity < -0.3: breakdown['Angry'] += 20
    if polarity > 0.4: breakdown['Happy'] += 20
    
    # Normalize breakdown
    total_b = sum(breakdown.values())
    if total_b < 100:
        breakdown['Neutral'] = 100 - total_b
    else:
        for k in breakdown:
            breakdown[k] = int((breakdown[k] / total_b) * 100)
        breakdown['Neutral'] = 100 - sum(breakdown.values())
    
    # Risk Score for non-spam
    risk_score = min(100, int((urgency_score * 0.5) + (breakdown.get('Angry', 0) * 0.5)))
    if risk_score < 5: risk_score = 5 # baseline
    
    # Recommendations
    action = "No immediate action required. Standard priority."
    if primary_emotion == 'Urgent': action = "Respond quickly. High urgency detected. Prioritize this email."
    elif primary_emotion == 'Angry': action = "Review carefully. Proceed with empathy to de-escalate the situation."
    elif primary_emotion == 'Professional': action = "Standard business response recommended. File accordingly."
    elif primary_emotion == 'Happy': action = "Acknowledge positively to maintain good relationship."
    
    return {
        'primary_emotion': primary_emotion,
        'urgency_score': urgency_score,
        'breakdown': breakdown,
        'polarity': round(polarity, 2),
        'subjectivity': round(subjectivity, 2),
        'risk_score': risk_score,
        'recommendation': action
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        email_text = request.form.get('email_text', '')
        
        if not email_text.strip():
            return render_template('index.html', error="Please enter some text to analyze.")
            
        cleaned_text = preprocess_text(email_text)
        
        # Predict spam
        if spam_model:
            try:
                prediction = spam_model.predict([cleaned_text])[0]
                if hasattr(spam_model, "predict_proba"):
                    probs = spam_model.predict_proba([cleaned_text])[0]
                    confidence = max(probs) * 100
                else:
                    confidence = 100.0
                is_spam = bool(prediction == 1)
            except Exception as e:
                return render_template('index.html', error=f"Error processing prediction: {e}")
        else:
            is_spam = False
            confidence = 85.5  # fallback mock confidence
            
        if is_spam:
            result = "Spam"
            analysis = None
            risk_score = min(99, int(confidence))
            recommendation = "Do not click any links. Delete or block the sender immediately."
        else:
            result = "Not Spam"
            analysis = analyze_emotion_and_tone(cleaned_text)
            risk_score = analysis['risk_score']
            recommendation = analysis['recommendation']
            
        return render_template('index.html', 
                               email_text=email_text,
                               result=result, 
                               analysis=analysis,
                               confidence=f"{confidence:.1f}",
                               risk_score=risk_score,
                               recommendation=recommendation)
                               
    # Sample placeholder email text
    placeholder = "Dear Customer, \n\nPlease review the attached invoice urgently to avoid account suspension.\n\nThanks,\nSupport Team"
    return render_template('index.html', placeholder=placeholder)

if __name__ == '__main__':
    app.run(debug=True)
