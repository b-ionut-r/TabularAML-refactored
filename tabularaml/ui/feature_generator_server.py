#!/usr/bin/env python3
"""
Complete Flask server with comprehensive FeatureGenerator controls and dual modes
"""

import os
import json
import time
import threading
import pickle
import queue
from collections import deque
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from pathlib import Path
import io
from sklearn.utils.multiclass import type_of_target

# Import our actual FeatureGenerator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from tabularaml.generate.features import FeatureGenerator
from tabularaml.configs.feature_gen import PRESET_PARAMS
from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS, PREDEFINED_CLS_SCORERS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tabularaml_feature_gen_complete'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='gevent',
    ping_interval=20,
    ping_timeout=1800,
)

KEEPALIVE_INTERVAL_SECONDS = 10
RECENT_LOG_LIMIT = 200
_event_pump_lock = threading.Lock()

# Global state
server_state = {
    'is_training': False,
    'trained_generator': None,
    'current_generation': 0,
    'total_generations': 0,
    'current_child_eval': 0,
    'total_child_eval': 0,
    'selected_count': 0,
    'best_score': 0.0,
    'total_features': 0,
    'stagnation_level': 'NONE',
    'strategy': 'normal',
    'stop_requested': False,
    'generator_thread': None,
    'training_started_at': None,
    'latest_results': None,
    'last_error': None,
    'recent_logs': deque(maxlen=RECENT_LOG_LIMIT),
    'last_emit_at': None,
    'last_client_sid': None,
    'event_pump_started': False,
    'event_queue': queue.Queue(),
}

def _update_status_snapshot(event_name, payload):
    """Keep the latest UI-visible state available across reconnects."""
    if event_name == 'log_update':
        message = payload.get('message')
        if message:
            server_state['recent_logs'].append(message)
    elif event_name == 'progress_update':
        if payload.get('type') == 'generation':
            server_state['current_generation'] = payload.get('current', server_state['current_generation'])
            server_state['total_generations'] = payload.get('total', server_state['total_generations'])
        elif payload.get('type') == 'child_eval':
            server_state['current_child_eval'] = payload.get('evaluated', 0)
            server_state['total_child_eval'] = payload.get('total', 0)
            server_state['selected_count'] = payload.get('selected', 0)
    elif event_name == 'score_update':
        server_state['best_score'] = payload.get('score', server_state['best_score'])
    elif event_name == 'feature_count_update':
        server_state['total_features'] = payload.get('count', server_state['total_features'])
    elif event_name == 'stagnation_update':
        server_state['stagnation_level'] = payload.get('level', server_state['stagnation_level'])
    elif event_name == 'strategy_update':
        server_state['strategy'] = payload.get('strategy', server_state['strategy'])
    elif event_name == 'generation_complete':
        server_state['latest_results'] = payload.get('results')
        server_state['is_training'] = False
    elif event_name == 'error':
        server_state['last_error'] = payload.get('message')
        server_state['is_training'] = False

def get_status_snapshot():
    started_at = server_state.get('training_started_at')
    elapsed_seconds = 0
    if server_state['is_training'] and started_at:
        elapsed_seconds = int(max(0, time.time() - started_at))

    return {
        'is_training': server_state['is_training'],
        'current_generation': server_state['current_generation'],
        'total_generations': server_state['total_generations'],
        'current_child_eval': server_state['current_child_eval'],
        'total_child_eval': server_state['total_child_eval'],
        'selected_count': server_state['selected_count'],
        'best_score': server_state['best_score'],
        'total_features': server_state['total_features'],
        'stagnation_level': server_state['stagnation_level'],
        'strategy': server_state['strategy'],
        'elapsed_seconds': elapsed_seconds,
        'results': server_state['latest_results'],
        'last_error': server_state['last_error'],
        'has_trained_generator': server_state['trained_generator'] is not None,
        'recent_logs': list(server_state['recent_logs']),
    }

def socket_event_pump():
    """Emit UI events from a Socket.IO-managed background task."""
    while True:
        emitted = False

        while True:
            try:
                event_name, payload = server_state['event_queue'].get_nowait()
            except queue.Empty:
                break

            try:
                sid = server_state.get('last_client_sid')
                if sid:
                    socketio.emit(event_name, payload, to=sid)
                else:
                    socketio.emit(event_name, payload)
                emitted = True
                socketio.sleep(0)
            except Exception as e:
                print(f"Socket event pump failed for {event_name}: {e}")

        if server_state['is_training']:
            now = time.time()
            last_emit_at = server_state.get('last_emit_at') or now
            if now - last_emit_at >= KEEPALIVE_INTERVAL_SECONDS:
                started_at = server_state.get('training_started_at') or now
                keepalive_payload = {
                    'elapsed_seconds': int(max(0, now - started_at)),
                    'current_generation': server_state.get('current_generation', 0),
                    'total_generations': server_state.get('total_generations', 0)
                }
                try:
                    sid = server_state.get('last_client_sid')
                    if sid:
                        socketio.emit('keepalive', keepalive_payload, to=sid)
                    else:
                        socketio.emit('keepalive', keepalive_payload)
                    server_state['last_emit_at'] = now
                    emitted = True
                    socketio.sleep(0)
                except Exception as e:
                    print(f"Keepalive emit failed: {e}")

        socketio.sleep(0.25 if emitted else 1.0)

def ensure_event_pump():
    if server_state['event_pump_started']:
        return

    with _event_pump_lock:
        if server_state['event_pump_started']:
            return

        socketio.start_background_task(socket_event_pump)
        server_state['event_pump_started'] = True

def queue_socket_event(event_name, payload):
    normalized_payload = dict(payload)
    _update_status_snapshot(event_name, normalized_payload)
    if event_name != 'keepalive':
        server_state['last_emit_at'] = time.time()
    try:
        sid = server_state.get('last_client_sid')
        if sid:
            socketio.emit(event_name, normalized_payload, to=sid)
        else:
            socketio.emit(event_name, normalized_payload)
    except Exception as e:
        print(f"Direct emit failed for {event_name}, queueing fallback: {e}")
        ensure_event_pump()
        server_state['event_queue'].put((event_name, normalized_payload))

class ComprehensiveFeatureGenerator(FeatureGenerator):
    """Enhanced FeatureGenerator with comprehensive progress tracking"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_generation = 0
        self.stop_requested = False
        
    def _log(self, message):
        """Override logging to emit real log messages"""
        # Call original logging (but suppress file output if log_file is None)
        if hasattr(self, 'log_file') and self.log_file:
            super()._log(message)
        else:
            # Just print to console
            print(message)
        
        # Emit through the background pump so cross-thread delivery stays reliable.
        queue_socket_event('log_update', {
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
                    queue_socket_event('progress_update', {
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
                        score_str = parts[1].split()[0].replace(",", "").rstrip(".")
                        score = float(score_str)
                elif "Best " in message and ":" in message:
                    # Extract score from messages like "Best rmse: 0.12345"
                    parts = message.split("Best ")[1].split(": ")
                    if len(parts) >= 2:
                        score_str = parts[1].split()[0].replace(",", "").rstrip(".")
                        score = float(score_str)
                
                if score is not None:
                    queue_socket_event('score_update', {'score': score})
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
                        queue_socket_event('feature_count_update', {'count': total_features})
                        print(f"✨ Total features: {total_features}")
            except Exception as e:
                print(f"Error parsing features: {e}")
        
        # Parse stagnation level
        if "Status:" in message:
            try:
                status_part = message.split("Status: ")[1].split(",")[0].strip()
                queue_socket_event('stagnation_update', {'level': status_part})
                print(f"⚠️ Stagnation: {status_part}")
            except Exception as e:
                print(f"Error parsing stagnation: {e}")
        
        # Parse strategy changes
        if "Creative HM" in message or "hopeful monster" in message.lower():
            queue_socket_event('strategy_update', {'strategy': 'hopeful_monster'})
            print("🎯 Strategy: Hopeful Monster")
        elif "beam search" in message.lower():
            queue_socket_event('strategy_update', {'strategy': 'beam_search'})
            print("🎯 Strategy: Beam Search")
        elif message.startswith("Gen ") and "Added" in message:
            queue_socket_event('strategy_update', {'strategy': 'normal'})
    
    def _select_elites(self, batch, n, X, y, callback=None):
        """Override to capture child evaluation progress"""
        
        def progress_callback(evaluated_count, selected_count, force_complete=False):
            # Emit real-time child evaluation progress
            queue_socket_event('progress_update', {
                'type': 'child_eval',
                'evaluated': evaluated_count,
                'total': len(batch),
                'selected': selected_count
            })
            if force_complete or evaluated_count <= 3 or evaluated_count % 10 == 0:
                print(f"📶 Child progress emitted: {evaluated_count}/{len(batch)} selected={selected_count}")
            
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

    # Serve as bytes to avoid platform default text-decoding issues (cp1252 on Windows).
    return send_file(ui_file_path, mimetype='text/html')

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


@app.route('/get_metric_options', methods=['GET'])
def get_metric_options():
    """Expose available metrics for regression and classification tasks."""
    try:
        # Human-friendly labels for the UI while keeping scorer keys stable.
        reg_labels = {
            'rmse': 'RMSE (Root Mean Squared Error)',
            'mae': 'MAE (Mean Absolute Error)',
            'mse': 'MSE (Mean Squared Error)',
            'r2': 'R2 (Coefficient of Determination)'
        }
        cls_labels = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1 Score',
            'binary_crossentropy': 'Binary Cross-Entropy (Log Loss)',
            'categorical_crossentropy': 'Categorical Cross-Entropy (Log Loss)',
            'binary_roc_auc': 'ROC AUC (Binary)',
            'categorical_roc_auc': 'ROC AUC (Multiclass OVR)'
        }

        regression = [
            {'value': key, 'label': reg_labels.get(key, key)}
            for key in PREDEFINED_REG_SCORERS.keys()
        ]
        classification = [
            {'value': key, 'label': cls_labels.get(key, key)}
            for key in PREDEFINED_CLS_SCORERS.keys()
        ]

        return jsonify({
            'regression': regression,
            'classification': classification
        })
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
        ensure_event_pump()

        # Get basic parameters
        file = request.files.get('dataset')
        target = request.form.get('target')
        task = request.form.get('task', 'auto')
        metric = request.form.get('metric', 'auto')
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

        # Resolve task if not explicitly set so metric validation can be deterministic.
        if task == 'auto':
            inferred_task = 'regression' if type_of_target(y) == 'continuous' else 'classification'
        else:
            inferred_task = task

        scorer = None
        if metric and metric != 'auto':
            if inferred_task == 'regression':
                scorer = PREDEFINED_REG_SCORERS.get(metric)
            else:
                scorer = PREDEFINED_CLS_SCORERS.get(metric)

            if scorer is None and inferred_task == 'classification':
                # Soft compatibility mapping for binary vs multiclass-specific classification metrics.
                n_classes = int(y.nunique(dropna=True))
                if metric == 'binary_crossentropy' and n_classes > 2:
                    scorer = PREDEFINED_CLS_SCORERS.get('categorical_crossentropy')
                    print("ℹ️ Switched metric binary_crossentropy -> categorical_crossentropy for multiclass target")
                elif metric == 'categorical_crossentropy' and n_classes == 2:
                    scorer = PREDEFINED_CLS_SCORERS.get('binary_crossentropy')
                    print("ℹ️ Switched metric categorical_crossentropy -> binary_crossentropy for binary target")
                elif metric == 'binary_roc_auc' and n_classes > 2:
                    scorer = PREDEFINED_CLS_SCORERS.get('categorical_roc_auc')
                    print("ℹ️ Switched metric binary_roc_auc -> categorical_roc_auc for multiclass target")
                elif metric == 'categorical_roc_auc' and n_classes == 2:
                    scorer = PREDEFINED_CLS_SCORERS.get('binary_roc_auc')
                    print("ℹ️ Switched metric categorical_roc_auc -> binary_roc_auc for binary target")

            if scorer is None:
                return jsonify({'error': f'Invalid metric "{metric}" for task "{inferred_task}"'}), 400
        
        # Ensure y remains as pandas Series (don't convert to numpy array)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name=target) #
        
        print(f"📊 Dataset: {X.shape[0]} rows, {X.shape[1]} features")
        
        # Reset state
        server_state['is_training'] = True
        server_state['current_generation'] = 0
        server_state['total_generations'] = generations
        server_state['current_child_eval'] = 0
        server_state['total_child_eval'] = 0
        server_state['selected_count'] = 0
        server_state['best_score'] = 0.0
        server_state['total_features'] = X.shape[1]
        server_state['stagnation_level'] = 'NONE'
        server_state['strategy'] = 'normal'
        server_state['stop_requested'] = False
        server_state['training_started_at'] = time.time()
        server_state['latest_results'] = None
        server_state['last_error'] = None
        server_state['recent_logs'].clear()
        server_state['last_emit_at'] = time.time()
        
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
                if scorer is not None:
                    generator_params['scorer'] = scorer
                if max_new_feats:
                    generator_params['max_new_feats'] = int(max_new_feats)
                if time_budget:
                    generator_params['time_budget'] = int(time_budget) * 60  # Convert minutes to seconds
                if save_path:
                    generator_params['save_path'] = save_path
                
                # Create comprehensive generator
                generator = ComprehensiveFeatureGenerator(**generator_params)
                
                # Store generator in server state for stop functionality
                server_state['trained_generator'] = generator
                
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
                server_state['stop_requested'] = False
                server_state['training_started_at'] = None
                
                # Auto-save if path provided
                if save_path:
                    try:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        # Safely serialize without UI/socket references
                        try:
                            from tabularaml.generate.features import FeatureGenerator as _BaseFG
                            orig_cls = generator.__class__
                            generator.__class__ = _BaseFG
                            # Bind base _log to avoid capturing SocketIO in closures
                            try:
                                generator._log = _BaseFG._log.__get__(generator, _BaseFG)
                            except Exception:
                                pass
                            _BaseFG.save(generator, save_path)
                        finally:
                            # Restore original class regardless of save outcome
                            generator.__class__ = orig_cls
                        print(f"💾 Auto-saved to {save_path}")
                    except Exception as e:
                        print(f"❌ Auto-save failed: {e}")
                
                # Emit completion with generator data
                queue_socket_event('generation_complete', {
                    'results': results,
                    'generator_data': True  # Indicates generator is available for saving
                })
                print("🎉 Emitted comprehensive generation complete")
                
            except Exception as e:
                print(f"❌ Comprehensive generation error: {e}")
                import traceback
                traceback.print_exc()
                server_state['is_training'] = False
                server_state['stop_requested'] = False
                server_state['training_started_at'] = None
                queue_socket_event('error', {'message': str(e)})
        
        # Start comprehensive generation thread
        thread = threading.Thread(target=run_comprehensive_generation)
        thread.daemon = True
        thread.start()
        
        return jsonify({'status': 'Comprehensive generation started with all parameters'})
        
    except Exception as e:
        print(f"❌ Start generation error: {e}")
        server_state['is_training'] = False
        return jsonify({'error': str(e)}), 500

@app.route('/stop_generation', methods=['POST'])
def stop_generation():
    """Stop the running feature generation"""
    try:
        if not server_state['is_training']:
            return jsonify({'error': 'No generation is currently running'}), 400
        
        print("🛑 Stop generation requested")
        server_state['stop_requested'] = True
        
        # Signal the ComprehensiveFeatureGenerator to stop
        if server_state['trained_generator']:
            server_state['trained_generator'].stop_requested = True
        
        queue_socket_event('status_update', {'message': 'Stopping generation...'})
        return jsonify({'status': 'Stop request sent'})
        
    except Exception as e:
        print(f"❌ Stop generation error: {e}")
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
        
        # Save the generator using a plain FeatureGenerator snapshot to avoid pickling UI locks
        gen = server_state['trained_generator']
        from tabularaml.generate.features import FeatureGenerator as _BaseFG
        try:
            orig_cls = gen.__class__
            gen.__class__ = _BaseFG
            try:
                gen._log = _BaseFG._log.__get__(gen, _BaseFG)
            except Exception:
                pass
            _BaseFG.save(gen, save_path)
        finally:
            gen.__class__ = orig_cls
        
        queue_socket_event('save_complete', {'path': save_path})
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
        
        queue_socket_event('transform_complete', {
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
    ensure_event_pump()
    server_state['last_client_sid'] = request.sid
    print('🔌 Client connected for comprehensive feature generation')
    emit('status_snapshot', get_status_snapshot())

@socketio.on('disconnect')
def handle_disconnect():
    if server_state.get('last_client_sid') == request.sid:
        server_state['last_client_sid'] = None
    print('🔌 Client disconnected')

if __name__ == '__main__':
    print("🚀 Starting Comprehensive TabularAML Feature Generator Server")
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', '5000'))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'

    print(f"📱 Open http://localhost:{port} in your browser")
    print("🎛️ Features:")
    print("   • Train mode: Full parameter control + real progress tracking")
    print("   • Transform mode: Load saved generators + transform new data")
    print("   • Save/Load: Persistent generator storage")
    print("   • Beautiful UI: Comprehensive controls with tooltips")
    
    # Ensure directories exist
    os.makedirs("cache", exist_ok=True)
    os.makedirs("cache/logs", exist_ok=True)
    
    socketio.run(app, debug=debug, host=host, port=port)
