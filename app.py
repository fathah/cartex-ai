from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Load pre-trained embedding model (you can use other models like 'paraphrase-MiniLM-L6-v2')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Route to generate embedding for product
@app.route('/generate_embedding', methods=['POST'])
def generate_embedding():
    # Get product data from request
    data = request.get_json()
    name = data.get('name', '')
    description = data.get('description', '')
    category = data.get('category', '')

    # Combine product details into a single string
    combined_text = f"{name}. {description}. {category}"

    # Generate the embedding using the model
    embedding = model.encode(combined_text).tolist()  # Convert to list for JSON compatibility

    # Return the embedding as a JSON response
    return jsonify({
        'embedding': embedding,
        'message': 'Embedding generated successfully'
    })

# Start the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
