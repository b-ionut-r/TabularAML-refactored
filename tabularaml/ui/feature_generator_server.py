#!/usr/bin/env python3
"""
Complete Flask server with comprehensive FeatureGenerator controls and dual modes
"""

import os
import json
import time
import threading
import pickle
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from pathlib import Path
import io

# Import our actual FeatureGenerator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from tabularaml.generate.features import FeatureGenerator
from tabularaml.configs.feature_gen import PRESET_PARAMS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tabularaml_feature_gen_complete'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
server_state = {
    'is_training': False,
    'trained_generator': None,
    'current_generation': 0,
    'total_generations': 0
}

class ComprehensiveFeatureGenerator(FeatureGenerator):
    """Enhanced FeatureGenerator with comprehensive progress tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_generation = 0
        
    def _log(self, message):
        """Override logging to emit real log messages"""
        # Call original logging (but suppress file output if log_file is None)
        if hasattr(self, 'log_file') and self.log_file:
            super()._log(message)
        else:
            # Just print to console
            print(message)
        
        # Emit to UI
        socketio.emit('log_update', {
            'message': f'[{datetime.now().strftime("%H:%M:%S")}] {message}'
        })
        
        # Parse generation info from log messages
        if message.startswith("Gen ") and ":" in message:
            try:
                gen_part = message.split(":")[0].replace("Gen ", "").strip()
                if gen_part.isdigit():
                    gen_num = int(gen_part)
                    self.current_generation = gen_num
                    server_state['current_generation'] = gen_num
                    socketio.emit('progress_update', {
                        'type': 'generation',
                        'current': gen_num,
                        'total': self.n_generations
                    })
                    print(f"📊 Generation progress: {gen_num}/{self.n_generations}")
            except Exception as e:
                print(f"Error parsing generation: {e}")
        
        # Parse score updates
        if ("Val " in message and "=" in message) or ("Best " in message and ":" in message):
            try:
                score = None
                if "Val " in message and "=" in message:
                    # Extract score from messages like "Val rmse=0.12345"
                    parts = message.split("Val ")[1].split("=")
                    if len(parts) >= 2:
                        score_str = parts[1].split()[0].replace(",", "")
                        score = float(score_str)
                elif "Best " in message and ":" in message:
                    # Extract score from messages like "Best rmse: 0.12345"
                    parts = message.split("Best ")[1].split(": ")
                    if len(parts) >= 2:
                        score_str = parts[1].split()[0].replace(",", "")
                        score = float(score_str)
                
                if score is not None:
                    socketio.emit('score_update', {'score': score})
                    print(f"📈 Score update: {score}")
            except Exception as e:
                print(f"Error parsing score: {e}")
        
        # Parse feature counts
        if "Added" in message and "features" in message:
            try:
                # Extract from messages like "Gen 1: Added 3 features, 15 total"
                if "total" in message:
                    total_part = message.split("total")[0].split()[-1]
                    if total_part.isdigit():
                        total_features = int(total_part)
                        socketio.emit('feature_count_update', {'count': total_features})
                        print(f"✨ Total features: {total_features}")
            except Exception as e:
                print(f"Error parsing features: {e}")
        
        # Parse stagnation level
        if "Status:" in message:
            try:
                status_part = message.split("Status: ")[1].split(",")[0].strip()
                socketio.emit('stagnation_update', {'level': status_part})
                print(f"⚠️ Stagnation: {status_part}")
            except Exception as e:
                print(f"Error parsing stagnation: {e}")
        
        # Parse strategy changes
        if "Creative HM" in message or "hopeful monster" in message.lower():
            socketio.emit('strategy_update', {'strategy': 'hopeful_monster'})
            print("🎯 Strategy: Hopeful Monster")
        elif "beam search" in message.lower():
            socketio.emit('strategy_update', {'strategy': 'beam_search'})
            print("🎯 Strategy: Beam Search")
        elif message.startswith("Gen ") and "Added" in message:
            socketio.emit('strategy_update', {'strategy': 'normal'})
    
    def _select_elites(self, batch, n, X, y, callback=None):
        """Override to capture child evaluation progress"""
        
        def progress_callback(evaluated_count, selected_count, force_complete=False):
            # Emit real-time child evaluation progress
            socketio.emit('progress_update', {
                'type': 'child_eval',
                'evaluated': evaluated_count,
                'total': len(batch),
                'selected': selected_count
            })
            
            # Call original callback if provided
            if callback:
                return callback(evaluated_count, selected_count, force_complete)
            return False
        
        # Call original method with our progress callback
        return super()._select_elites(batch, n, X, y, progress_callback)

@app.route('/')
def index():
    """Serve the comprehensive UI"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ui_file_path = os.path.join(script_dir, 'feature_generator_ui.html')
    
    with open(ui_file_path, 'r') as f:
        return f.read()

@app.route('/get_mode_presets', methods=['GET'])
def get_mode_presets():
    """Get mode preset configurations"""
    try:
        # Convert config to UI-friendly format with mapping
        ui_presets = {}
        for mode, params in PRESET_PARAMS.items():
            ui_presets[mode] = {
                'generations': params.get('n_generations', 15),
                'parents': params.get('n_parents', 40), 
                'children': params.get('n_children', 200),
                'early_stop_child': params.get('early_stopping_child_eval', 0.3),
                'early_stop_iter': params.get('early_stopping_iter', 0.4),
                'min_pct_gain': params.get('min_pct_gain', 0.001),
                'cv_folds': params.get('cv', 5),
                'time_budget': params.get('time_budget', '') if params.get('time_budget') else ''
            }
        
        # Add 'none' mode with defaults
        ui_presets['none'] = {
            'generations': 15,
            'parents': 40,
            'children': 200,
            'early_stop_child': 0.3,
            'early_stop_iter': 0.4,
            'min_pct_gain': 0.001,
            'cv_folds': 5,
            'time_budget': ''
        }
        
        return jsonify(ui_presets)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_columns', methods=['POST'])
def get_columns():
    """Get column names from uploaded dataset"""
    try:
        file = request.files['dataset']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file, nrows=0)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file, nrows=0)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
            
        return jsonify({'columns': list(df.columns)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start_generation', methods=['POST'])
def start_generation():
    """Start comprehensive feature generation with all parameters"""
    global server_state
    
    if server_state['is_training']:
        return jsonify({'error': 'Generation already running'}), 400
    
    try:
        # Get basic parameters
        file = request.files.get('dataset')
        target = request.form.get('target')
        task = request.form.get('task', 'auto')
        mode = request.form.get('mode', 'medium')
        
        # Helper function to parse parameters (empty = use default)
        def parse_param(key, default, param_type=int):
            value = request.form.get(key, '')
            if value == '' or value is None:
                return default
            try:
                return param_type(value)
            except:
                return default
        
        # Parse parameters - empty values will use constructor defaults (mode will override)
        generations = parse_param('generations', 15, int)
        parents = parse_param('parents', 40, int)
        children = parse_param('children', 200, int)
        min_pct_gain = parse_param('min_pct_gain', 0.001, float)
        early_stop_iter = parse_param('early_stop_iter', 0.4, float)
        early_stop_child = parse_param('early_stop_child', 0.3, float)
        cv_folds = parse_param('cv_folds', 5, int)
        
        # Handle optional parameters
        max_new_feats = request.form.get('max_new_feats', '')
        ranking_method = request.form.get('ranking_method', 'multi_criteria')
        time_budget = request.form.get('time_budget', '')
        save_path = request.form.get('save_path', '')
        use_gpu = request.form.get('use_gpu', 'false').lower() == 'true'
        adaptive = request.form.get('adaptive', 'false').lower() == 'true'
        
        print(f"🚀 Starting comprehensive generation with {generations} gens, {parents} parents, {children} children")
        print(f"📊 Advanced params: min_gain={min_pct_gain}, early_stop={early_stop_iter}, ranking={ranking_method}")
        
        if not file or not target:
            return jsonify({'error': 'Dataset and target column required'}), 400
        
        # Load dataset
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        if target not in df.columns:
            return jsonify({'error': f'Target column {target} not found'}), 400
        
        # Prepare data
        X = df.drop(columns=[target])
        y = df[target]
        
        # Ensure y remains as pandas Series (don't convert to numpy array)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=target)
        
        print(f"📊 Dataset: {X.shape[0]} rows, {X.shape[1]} features")
        
        # Reset state
        server_state['is_training'] = True
        server_state['current_generation'] = 0
        server_state['total_generations'] = generations
        
        # Start comprehensive generation in background thread
        def run_comprehensive_generation():
            try:
                print("🧠 Creating ComprehensiveFeatureGenerator with all parameters...")
                
                # Prepare parameters
                generator_params = {
                    'n_generations': generations,
                    'n_parents': parents,
                    'n_children': children,
                    'min_pct_gain': min_pct_gain,
                    'early_stopping_iter': early_stop_iter,
                    'early_stopping_child_eval': early_stop_child,
                    'ranking_method': ranking_method,
                    'cv': cv_folds,
                    'use_gpu': use_gpu,
                    'adaptive': adaptive,
                    'log_file': None  # We handle logging ourselves
                }
                
                # Add optional parameters
                if mode != 'auto' and mode != 'none':
                    generator_params['mode'] = mode
                if task != 'auto':
                    generator_params['task'] = task
                if max_new_feats:
                    generator_params['max_new_feats'] = int(max_new_feats)
                if time_budget:
                    generator_params['time_budget'] = int(time_budget) * 60  # Convert minutes to seconds
                if save_path:
                    generator_params['save_path'] = save_path
                
                # Create comprehensive generator
                generator = ComprehensiveFeatureGenerator(**generator_params)
                
                print("🚀 Running comprehensive FeatureGenerator.search()...")
                start_time = time.time()
                
                # This will trigger all the real progress updates!
                X_result, pipeline, generation_features, interactions = generator.search(X, y)
                
                end_time = time.time()
                print("✅ Comprehensive FeatureGenerator completed!")
                
                # Store trained generator
                server_state['trained_generator'] = generator
                
                # Calculate comprehensive results
                results = {
                    'total_time': round(end_time - start_time, 2),
                    'completed_gens': generator.current_generation,
                    'features_added': len(X_result.columns) - len(X.columns),
                    'initial_score': getattr(generator, 'initial_val_metric', 0.0),
                    'final_score': getattr(generator, 'final_metric', 0.0),
                    'improvement': getattr(generator, 'gain', 0.0),
                    'percent_gain': getattr(generator, 'pct_gain', 0.0) * 100,
                    'total_restarts': getattr(generator.adaptive_controller.state, 'total_restarts', 0) if hasattr(generator, 'adaptive_controller') else 0,
                    'best_generation': getattr(generator.state['best'], 'gen_num', 0) if hasattr(generator, 'state') else 0
                }
                
                server_state['is_training'] = False
                
                # Auto-save if path provided
                if save_path:
                    try:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        generator.save(save_path)
                        print(f"💾 Auto-saved to {save_path}")
                    except Exception as e:
                        print(f"❌ Auto-save failed: {e}")
                
                # Emit completion with generator data
                socketio.emit('generation_complete', {
                    'results': results,
                    'generator_data': True  # Indicates generator is available for saving
                })
                print("🎉 Emitted comprehensive generation complete")
                
            except Exception as e:
                print(f"❌ Comprehensive generation error: {e}")
                import traceback
                traceback.print_exc()
                server_state['is_training'] = False
                socketio.emit('error', {'message': str(e)})
        
        # Start comprehensive generation thread
        thread = threading.Thread(target=run_comprehensive_generation)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'Comprehensive generation started with all parameters'})
        
    except Exception as e:
        print(f"❌ Start generation error: {e}")
        server_state['is_training'] = False
        return jsonify({'error': str(e)}), 500

@app.route('/save_generator', methods=['POST'])
def save_generator():
    """Save the trained generator"""
    try:
        if not server_state['trained_generator']:
            return jsonify({'error': 'No trained generator available'}), 400
        
        data = request.get_json()
        save_path = data.get('save_path', 'cache/feature_generator.pkl')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save the generator
        server_state['trained_generator'].save(save_path)
        
        socketio.emit('save_complete', {'path': save_path})
        return jsonify({'status': 'Generator saved successfully', 'path': save_path})
        
    except Exception as e:
        print(f"❌ Save error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/load_generator', methods=['POST'])
def load_generator():
    """Load a saved generator for transformation"""
    try:
        file = request.files.get('generator_file')
        if not file:
            return jsonify({'error': 'No generator file provided'}), 400
        
        # Save temporarily and load
        temp_path = 'temp_generator.pkl'
        file.save(temp_path)
        
        # Load the generator
        loaded_generator = FeatureGenerator.load(temp_path)
        server_state['trained_generator'] = loaded_generator
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify({'status': 'Generator loaded successfully'})
        
    except Exception as e:
        print(f"❌ Load error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/transform_data', methods=['POST'])
def transform_data():
    """Transform data using loaded generator"""
    try:
        if not server_state['trained_generator']:
            return jsonify({'error': 'No generator loaded'}), 400
        
        file = request.files.get('dataset')
        if not file:
            return jsonify({'error': 'No dataset provided'}), 400
        
        # Load dataset
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        print(f"🔄 Transforming dataset: {df.shape[0]} rows, {df.shape[1]} features")
        
        # Transform data
        start_time = time.time()
        df_transformed = server_state['trained_generator'].transform(df)
        end_time = time.time()
        
        # Prepare results
        original_features = df.shape[1]
        transformed_features = df_transformed.shape[1]
        features_added = transformed_features - original_features
        transform_time = round(end_time - start_time, 2)
        
        # Convert to CSV string for download
        csv_buffer = io.StringIO()
        df_transformed.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        socketio.emit('transform_complete', {
            'original_features': original_features,
            'transformed_features': transformed_features,
            'features_added': features_added,
            'transform_time': transform_time,
            'transformed_data': csv_data
        })
        
        return jsonify({'status': 'Data transformed successfully'})
        
    except Exception as e:
        print(f"❌ Transform error: {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    print('🔌 Client connected for comprehensive feature generation')

@socketio.on('disconnect')
def handle_disconnect():
    print('🔌 Client disconnected')

if __name__ == '__main__':
    print("🚀 Starting Comprehensive TabularAML Feature Generator Server")
    print("📱 Open http://localhost:5000 in your browser")
    print("🎛️ Features:")
    print("   • Train mode: Full parameter control + real progress tracking")
    print("   • Transform mode: Load saved generators + transform new data")
    print("   • Save/Load: Persistent generator storage")
    print("   • Beautiful UI: Comprehensive controls with tooltips")
    
    # Ensure directories exist
    os.makedirs("cache", exist_ok=True)
    os.makedirs("cache/logs", exist_ok=True)
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)