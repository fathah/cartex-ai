import os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


app = Flask(__name__)

load_dotenv()
VALID_API_KEY = os.getenv('API_KEY')

model = SentenceTransformer('all-MiniLM-L6-v2')

# Route to generate embedding for product
@app.route('/generate_embedding', methods=['POST'])
def generate_embedding():
    try:
        api_key = request.headers.get('x-api-key')
        if api_key != VALID_API_KEY:
            return jsonify({'success': False, 'message': 'Invalid or missing API key'}), 403


        data = request.get_json()
        name = data.get('name', '')
        description = data.get('description', '')
        category = data.get('category', '')

        combined_text = f"{name}. {description}. {category}"

        embedding = model.encode(combined_text).tolist()

        # Return the embedding as a JSON response
        return jsonify({
            'success': True,
            'embedding': embedding,
        })
    except Exception as e:
        
        return jsonify({'success': False, 'message': str(e)}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
