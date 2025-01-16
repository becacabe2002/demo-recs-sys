from flask import Flask, request, jsonify, render_template
import pandas as pd
import json
import os
import time
from builders.funk_svd import FunkSVD
from builders.cluster_svd import ClusterAsUserSVD
from recs.funk_recs import FunkSVDRecs
from recs.cluster_recs import ClusterAsUserRecs
from utils.data_processing import process_ratings, get_cluster_data, clean_data

app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'saved_models'
for folder in [UPLOAD_FOLDER, MODEL_FOLDER, 
              os.path.join(MODEL_FOLDER, 'base_funk'),
              os.path.join(MODEL_FOLDER, 'cluster')]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Global variables
ratings_df = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model-status', methods=['GET'])
def check_model_status():
    base_path = os.path.join(MODEL_FOLDER, 'base_funk')
    cluster_path = os.path.join(MODEL_FOLDER, 'cluster')
    
    # Check base model files
    base_files = [
        'user_factors.json',
        'item_factors.json',
        'user_bias.data',
        'item_bias.data',
        'metadata.json'
    ]
    base_ready = all(os.path.exists(os.path.join(base_path, f)) for f in base_files)
    
    # Check cluster model files
    cluster_files = [
        os.path.join(cluster_path, 'funk_svd', 'user_factors.json'),
        os.path.join(cluster_path, 'funk_svd', 'item_factors.json'),
        os.path.join(cluster_path, 'funk_svd', 'user_bias.data'),
        os.path.join(cluster_path, 'funk_svd', 'item_bias.data'),
        os.path.join(cluster_path, 'funk_svd', 'metadata.json'),
        os.path.join(cluster_path, 'user_clusters.json'),
        os.path.join(cluster_path, 'kmeans.npy')
    ]
    cluster_ready = all(os.path.exists(f) for f in cluster_files)
    
    return jsonify({
        'base_model': base_ready,
        'cluster_model': cluster_ready
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Save file temporarily
        filepath = os.path.join(UPLOAD_FOLDER, 'ratings.csv')
        file.save(filepath)
        
        # Load ratings exactly as in benchmark implementation
        global ratings_df
        ratings_df = pd.read_csv(filepath)
        ratings_df = ratings_df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
        
        # Clean data and get stats
        ratings_df = clean_data(ratings_df, min_ratings=20)
        
        # Get statistics using renamed columns
        stats = {
            'num_users': len(ratings_df['user_id'].unique()),
            'num_items': len(ratings_df['movie_id'].unique()),
            'num_ratings': len(ratings_df)
        }
        
        # Process and store data
        process_ratings(ratings_df, noise_scale=0.1)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'statistics': stats
        })
    except Exception as e:
        print(f"Upload error: {str(e)}")  # Add debug print
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_models():
    if ratings_df is None:
        return jsonify({'error': 'No data uploaded yet'}), 400
        
    model_type = request.json.get('type')
    if model_type not in ['base', 'cluster']:
        return jsonify({'error': 'Invalid model type specified'}), 400
        
    try:
        start_time = time.time()
        
        if model_type == 'base':
            # Train basic Funk MF
            model = FunkSVD(
                save_path=os.path.join(MODEL_FOLDER, 'base_funk'),
                learning_rate=0.002,
                bias_learning_rate=0.005,
                bias_reg=0.001,
                max_iterations=50
            )
            model.train(ratings_df, k=20)
        else:
            # Train Privacy-Enhanced Cluster Model
            cluster_ratings, _ = get_cluster_data()
            model = ClusterAsUserSVD(
                save_path=os.path.join(MODEL_FOLDER, 'cluster'),
                learning_rate=0.002,
                max_iterations=50,
                contribution_prob=0.90  # % random gradient to be kept
            )
            model.train(cluster_ratings, k=20)
            
        training_time = time.time() - start_time
        
        return jsonify({
            'message': f'{model_type} model trained successfully',
            'training_time': training_time
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    if ratings_df is None:
        return jsonify({'error': 'No data uploaded yet'}), 400
        
    user_id = request.json.get('user_id')
    model_type = request.json.get('type')
    
    if not user_id:
        return jsonify({'error': 'No user ID provided'}), 400
    
    try:
        start_time = time.time()
        
        if model_type == 'base':
            base_path = os.path.join(MODEL_FOLDER, 'base_funk')
            # Check if model files exist
            required_files = [
                'user_factors.json',
                'item_factors.json',
                'user_bias.data',
                'item_bias.data',
                'metadata.json'
            ]
            if not all(os.path.exists(os.path.join(base_path, f)) for f in required_files):
                return jsonify({'error': 'Base model not trained yet'}), 400
                
            recommender = FunkSVDRecs(
                ratings_df,
                save_path=base_path
            )
            recommendations = recommender.recommend_items(user_id, num=10)
        else:
            cluster_path = os.path.join(MODEL_FOLDER, 'cluster')
            # Check if model and cluster files exist
            required_files = [
                os.path.join(cluster_path, 'funk_svd', 'user_factors.json'),
                os.path.join(cluster_path, 'funk_svd', 'item_factors.json'),
                os.path.join(cluster_path, 'funk_svd', 'user_bias.data'),
                os.path.join(cluster_path, 'funk_svd', 'item_bias.data'),
                os.path.join(cluster_path, 'funk_svd', 'metadata.json'),
                os.path.join(cluster_path, 'user_clusters.json'),
                os.path.join(cluster_path, 'kmeans.npy')
            ]
            if not all(os.path.exists(f) for f in required_files):
                return jsonify({'error': 'Cluster model not trained yet'}), 400
                
            recommender = ClusterAsUserRecs(
                ratings_df,
                save_path=cluster_path
            )
            recommendations = recommender.recommend_items(user_id, num=10)
            
        recommendation_time = time.time() - start_time
        
        return jsonify({
            'recommendations': recommendations,
            'recommendation_time': recommendation_time
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
