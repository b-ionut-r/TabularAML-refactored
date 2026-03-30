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
import ast
from collections import deque
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, send_file
from flask_socketio import SocketIO, emit
import pandas as pd
import numpy as np
from pathlib import Path
import io
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import GroupKFold, TimeSeriesSplit

# Import our actual FeatureGenerator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from tabularaml.generate.features import FeatureGenerator
from tabularaml.configs.feature_gen import PRESET_PARAMS
from tabularaml.eval.scorers import PREDEFINED_REG_SCORERS, PREDEFINED_CLS_SCORERS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tabularaml_feature_gen_complete'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB max file size
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
    'generated_features': [],
    'transformed_train': None,
    'transformed_test': None,
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
    elif event_name == 'generated_features_update':
        server_state['generated_features'] = payload.get('features', server_state['generated_features'])
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
        'generated_features': list(server_state['generated_features']),
    }

def _emit_live_socket_event(event_name, payload):
    sid = server_state.get('last_client_sid')
    if sid:
        socketio.emit(event_name, payload, to=sid, namespace='/')
    else:
        socketio.emit(event_name, payload, namespace='/')
    socketio.sleep(0)

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
                _emit_live_socket_event(event_name, payload)
                emitted = True
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
                    _emit_live_socket_event('keepalive', keepalive_payload)
                    server_state['last_emit_at'] = now
                    emitted = True
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
        _emit_live_socket_event(event_name, normalized_payload)
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
        # Note: Do not prepend timestamp here; let the UI handle timestamps so they use a consistent format.
        queue_socket_event('log_update', {
            'message': message
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
        elif message.startswith("Gen ") and "Added" in message:
            queue_socket_event('strategy_update', {'strategy': 'normal'})

        stripped_message = message.strip()
        feature_prefixes = {
            "Simple:": "simple",
            "Target encoded:": "_target_enc",
            "Count encoded:": "_count_enc",
            "Freq encoded:": "_freq_enc",
            "New simple:": "simple",
            "New target:": "_target_enc",
            "New count:": "_count_enc",
            "New freq:": "_freq_enc",
        }
        for prefix, encoding_type in feature_prefixes.items():
            if stripped_message.startswith(prefix):
                try:
                    parsed = ast.literal_eval(stripped_message.split(":", 1)[1].strip())
                    if isinstance(parsed, str):
                        parsed = [parsed]
                    elif isinstance(parsed, set):
                        parsed = sorted(parsed)
                    elif not isinstance(parsed, (list, tuple)):
                        parsed = []

                    generated_features = list(server_state.get('generated_features', []))
                    for feature_name in parsed:
                        if isinstance(feature_name, str):
                            # Add encoding type suffix to categorical encoded features (except simple)
                            annotated_name = feature_name
                            if encoding_type != "simple" and not feature_name.endswith(encoding_type):
                                annotated_name = f"{feature_name}{encoding_type}"
                                
                            if annotated_name not in generated_features:
                                generated_features.append(annotated_name)

                    queue_socket_event('generated_features_update', {'features': generated_features})
                except Exception as e:
                    print(f"Error parsing generated features: {e}")
                break
    
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
                'time_budget': params.get('time_budget', '') if params.get('time_budget') else '',
                'search_sample_size': params.get('search_sample_size', '')
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
            'time_budget': '',
            'search_sample_size': ''
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
            'rmsle': 'RMSLE (Root Mean Squared Logarithmic Error)',
            'mae': 'MAE (Mean Absolute Error)',
            'mse': 'MSE (Mean Squared Error)',
            'r2': 'R2 (Coefficient of Determination)',
            'pearson': 'Pearson Correlation'
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

def _load_dataframe(file=None, file_path=None, nrows=None):
    """Load a DataFrame from either an uploaded file object or a server-side file path."""
    if file_path:
        file_path = file_path.strip()
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f'File not found: {file_path}')
        ext = os.path.splitext(file_path)[1].lower()
    elif file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
    else:
        raise ValueError('No dataset provided')

    source = file_path if file_path else file

    if ext == '.csv':
        return pd.read_csv(source, nrows=nrows)
    elif ext == '.parquet':
        if nrows == 0:
            import pyarrow.parquet as pq
            schema = pq.read_schema(source)
            return pd.DataFrame(columns=schema.names)
        df = pd.read_parquet(source)
        return df.iloc[:nrows] if nrows else df
    elif ext == '.json':
        return pd.read_json(source, nrows=nrows)
    else:
        raise ValueError(f'Unsupported file format: {ext}')


@app.route('/get_columns', methods=['POST'])
def get_columns():
    """Get column names from uploaded dataset or server-side path"""
    try:
        file = request.files.get('dataset')
        file_path = request.form.get('dataset_path', '').strip()
        df = _load_dataframe(file=file, file_path=file_path or None, nrows=0)
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
        file_path = request.form.get('dataset_path', '').strip() or None
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
                if param_type == int:
                    return int(float(value))  # handles "1.0", "5.0" etc.
                if param_type == float and '.' not in str(value):
                    return int(value)
                return param_type(value)
            except:
                return default
        
        # Parse parameters - store raw strings so we can tell blank from default
        generations = request.form.get('generations', '').strip()
        parents = request.form.get('parents', '').strip()
        children = request.form.get('children', '').strip()
        min_pct_gain = request.form.get('min_pct_gain', '').strip()
        early_stop_iter = request.form.get('early_stop_iter', '').strip()
        early_stop_child = request.form.get('early_stop_child', '').strip()
        cv_folds = parse_param('cv_folds', 5, int)
        cv_type = request.form.get('cv_type', 'kfold')
        group_col = request.form.get('group_col', '').strip()
        cv_gap = parse_param('cv_gap', 0, int)

        # Handle optional parameters
        max_new_feats = request.form.get('max_new_feats', '')
        ranking_method = request.form.get('ranking_method', 'multi_criteria')
        time_budget = request.form.get('time_budget', '')
        save_path = request.form.get('save_path', '')
        search_sample_size = request.form.get('search_sample_size', '')
        use_gpu = request.form.get('use_gpu', 'false').lower() == 'true'
        adaptive = request.form.get('adaptive', 'false').lower() == 'true'
        
        print(f"🚀 Starting generation — overrides: gens={generations or 'mode'}, parents={parents or 'mode'}, children={children or 'mode'}")
        print(f"📊 Advanced params: min_gain={min_pct_gain or 'mode'}, early_stop={early_stop_iter or 'mode'}, ranking={ranking_method}")
        
        if not (file or file_path) or not target:
            return jsonify({'error': 'Dataset and target column required'}), 400

        try:
            df = _load_dataframe(file=file, file_path=file_path)
        except (FileNotFoundError, ValueError) as e:
            return jsonify({'error': str(e)}), 400
        
        if target not in df.columns:
            return jsonify({'error': f'Target column {target} not found'}), 400
        
        # Prepare data
        X = df.drop(columns=[target])
        y = df[target]

        # # Prevent leakage by dropping the group/time column from predictors if used
        # if group_col and group_col in X.columns:
        #     print(f"🗑️ Dropping splitting column '{group_col}' from features to prevent leakage.")
        #     X = X.drop(columns=[group_col])

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
        
        # Build CV strategy and extract groups
        groups = None
        cv_obj = cv_folds  # default: integer (KFold/StratifiedKFold chosen in cv.py)
        if cv_type == 'groupfold':
            if not group_col:
                return jsonify({'error': 'Group column is required for GroupKFold'}), 400
            if group_col not in df.columns:
                return jsonify({'error': f'Group column "{group_col}" not found in dataset'}), 400
            groups = df[group_col].values
            cv_obj = GroupKFold(n_splits=cv_folds)
            print(f"📊 GroupKFold: {cv_folds} splits on column '{group_col}' ({len(np.unique(groups))} unique groups)")

        elif cv_type == 'timeseries':
            if not group_col:
                return jsonify({'error': 'Time column is required for TimeSeriesSplit'}), 400
            if group_col not in df.columns:
                return jsonify({'error': f'Time column "{group_col}" not found in dataset'}), 400
            groups = df[group_col].values
            unique_periods = np.sort(np.unique(groups))
            tss = TimeSeriesSplit(n_splits=cv_folds, gap=cv_gap)

            class _PurgedTimeSeriesSplit:
                """Wraps TimeSeriesSplit to work on period-level then mask rows."""
                def __init__(self, tss, unique_periods, groups):
                    self._tss = tss
                    self._periods = unique_periods
                    self._groups = groups
                    self.n_splits = tss.n_splits
                def split(self, X, y=None, groups=None):
                    # Use groups passed at split-time (may be subsampled), else fall back
                    g = groups if groups is not None else self._groups
                    unique_periods = np.sort(np.unique(g))
                    for tr_p_idx, val_p_idx in self._tss.split(unique_periods):
                        tr_periods  = unique_periods[tr_p_idx]
                        val_periods = unique_periods[val_p_idx]
                        tr_mask  = np.isin(g, tr_periods)
                        val_mask = np.isin(g, val_periods)
                        yield np.where(tr_mask)[0], np.where(val_mask)[0]
                def get_n_splits(self, X=None, y=None, groups=None):
                    return self._tss.get_n_splits()

            cv_obj = _PurgedTimeSeriesSplit(tss, unique_periods, groups)
            print(f"📊 TimeSeriesSplit: {cv_folds} splits, gap={cv_gap} on column '{group_col}' ({len(unique_periods)} unique periods)")

        elif cv_type == 'custom':
            splitter_file = request.files.get('custom_splitter_file')
            if not splitter_file:
                return jsonify({'error': 'No splitter file uploaded for Custom CV'}), 400
            splitter_code = splitter_file.read().decode('utf-8')
            ns = {}
            try:
                exec(compile(splitter_code, splitter_file.filename, 'exec'), ns)
            except Exception as e:
                return jsonify({'error': f'Error executing splitter file: {e}'}), 400
            if 'get_splitter' in ns and callable(ns['get_splitter']):
                try:
                    cv_obj = ns['get_splitter'](cv_folds)
                except Exception as e:
                    return jsonify({'error': f'get_splitter({cv_folds}) raised: {e}'}), 400
            elif 'splitter' in ns:
                cv_obj = ns['splitter']
            else:
                return jsonify({'error': 'Splitter file must define a splitter variable or get_splitter(n_splits) function'}), 400
            print(f"📊 Custom splitter loaded from '{splitter_file.filename}': {type(cv_obj).__name__}")

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
        # Don't clear generated_features - keep them persistent across runs
        # server_state['generated_features'] = []  # Features now persist after search ends
        
        # Start comprehensive generation in background thread
        def run_comprehensive_generation():
            try:
                print("🧠 Creating ComprehensiveFeatureGenerator with all parameters...")
                
                # Prepare parameters — only include overrides the user explicitly set
                generator_params = {
                    'ranking_method': ranking_method,
                    'cv': cv_obj,
                    'groups': groups,
                    'use_gpu': use_gpu,
                    'adaptive': adaptive,
                    'log_file': None
                }
                if generations:
                    generator_params['n_generations'] = int(float(generations))
                if parents:
                    generator_params['n_parents'] = int(float(parents))
                if children:
                    generator_params['n_children'] = int(float(children))
                if min_pct_gain:
                    generator_params['min_pct_gain'] = float(min_pct_gain)
                if early_stop_iter:
                    generator_params['early_stopping_iter'] = float(early_stop_iter) if '.' in early_stop_iter else int(early_stop_iter)
                if early_stop_child:
                    generator_params['early_stopping_child_eval'] = float(early_stop_child) if '.' in early_stop_child else int(early_stop_child)
                
                # Add optional parameters
                if mode != 'auto' and mode != 'none':
                    generator_params['mode'] = mode
                if task != 'auto':
                    generator_params['task'] = task
                if scorer is not None:
                    generator_params['scorer'] = scorer
                if max_new_feats:
                    generator_params['max_new_feats'] = (float(max_new_feats)
                                                         if '.' in str(max_new_feats)
                                                         else int(max_new_feats))
                if time_budget:
                    generator_params['time_budget'] = int(time_budget) * 60  # Convert minutes to seconds
                if save_path:
                    generator_params['save_path'] = save_path
                if search_sample_size:
                    generator_params['search_sample_size'] = int(search_sample_size)
                
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
                # Extract strategy success rates
                adaptive_ctrl = getattr(generator, 'adaptive_controller', None)
                if adaptive_ctrl:
                    normal_success = adaptive_ctrl.strategy_success.get('normal', 0)
                    hopeful_success = adaptive_ctrl.strategy_success.get('hopeful_monster', 0)

                    normal_attempts = adaptive_ctrl.strategy_attempts.get('normal', 1)
                    hopeful_attempts = adaptive_ctrl.strategy_attempts.get('hopeful_monster', 1)

                    normal_rate = (normal_success / normal_attempts * 100) if normal_attempts > 0 else 0.0
                    hopeful_rate = (hopeful_success / hopeful_attempts * 100) if hopeful_attempts > 0 else 0.0
                else:
                    normal_rate, hopeful_rate = 0.0, 0.0
                
                results = {
                    'total_time': round(end_time - start_time, 2),
                    'completed_gens': generator.current_generation,
                    'features_added': len(X_result.columns) - len(X.columns),
                    'initial_score': getattr(generator, 'initial_val_metric', 0.0),
                    'final_score': getattr(generator, 'final_metric', 0.0),
                    'improvement': getattr(generator, 'gain', 0.0),
                    'percent_gain': getattr(generator, 'pct_gain', 0.0) * 100,
                    'total_restarts': getattr(generator.adaptive_controller.state, 'total_restarts', 0) if hasattr(generator, 'adaptive_controller') else 0,
                    'best_generation': getattr(generator.state['best'], 'gen_num', 0) if hasattr(generator, 'state') else 0,
                    'normal_strategy_success': round(normal_rate, 2),
                    'hopeful_monster_success': round(hopeful_rate, 2)
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
        
        # Run generation inside the Socket.IO async runtime so emits flush correctly.
        thread = socketio.start_background_task(run_comprehensive_generation)
        server_state['generator_thread'] = thread
        
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
        
        data = request.get_json(silent=True) or {}
        save_path = data.get('save_path', 'cache/feature_generator.pkl')
        should_download = bool(data.get('download', False))
        
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
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
        if should_download:
            download_name = os.path.basename(save_path) or 'feature_generator.pkl'
            return send_file(
                os.path.abspath(save_path),
                as_attachment=True,
                download_name=download_name,
                mimetype='application/octet-stream'
            )
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
        
        train_file = request.files.get('train_dataset')
        test_file = request.files.get('test_dataset')
        
        if not train_file and not test_file:
            # Fallback to older 'dataset' name just in case
            train_file = request.files.get('dataset')
            if not train_file:
                return jsonify({'error': 'No dataset provided'}), 400
        
        def load_df(f):
            return _load_dataframe(file=f)
        
        df_train = None
        df_test = None

        if train_file:
            df_train = load_df(train_file)
        if test_file:
            df_test = load_df(test_file)
            
        start_time = time.time()
        
        # Fit transform on train if present, else just transform
        # And transform on test if present
        
        df_train_transformed = None
        df_test_transformed = None
        
        if df_train is not None:
            print(f"🔄 Fit-Transforming train dataset: {df_train.shape[0]} rows, {df_train.shape[1]} features")
            # Usually target 'y' might be needed if they used TargetEncoders, 
            # but we'll call fit_transform(X) and rely on any fallback Behavior or it just fits imputations
            df_train_transformed = server_state['trained_generator'].fit_transform(df_train)
            
        if df_test is not None:
            print(f"🔄 Transforming test dataset: {df_test.shape[0]} rows, {df_test.shape[1]} features")
            df_test_transformed = server_state['trained_generator'].transform(df_test)
            
        end_time = time.time()
        transform_time = round(end_time - start_time, 2)
        
        # Store transformed DataFrames server-side for download; send only metadata via socket
        server_state['transformed_train'] = df_train_transformed
        server_state['transformed_test'] = df_test_transformed

        result_payload = {
            'transform_time': transform_time,
            'has_train': df_train_transformed is not None,
            'has_test': df_test_transformed is not None
        }

        if df_train_transformed is not None:
            result_payload['train_original_features'] = df_train.shape[1]
            result_payload['train_transformed_features'] = df_train_transformed.shape[1]
            result_payload['train_features_added'] = df_train_transformed.shape[1] - df_train.shape[1]

        if df_test_transformed is not None:
            result_payload['test_original_features'] = df_test.shape[1]
            result_payload['test_transformed_features'] = df_test_transformed.shape[1]
            result_payload['test_features_added'] = df_test_transformed.shape[1] - df_test.shape[1]
        
        queue_socket_event('transform_complete', result_payload)
        
        return jsonify({'status': 'Data transformed successfully'})
        
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        print(f"❌ Transform error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_transformed_data/<dtype>', methods=['GET'])
def get_transformed_data(dtype):
    """Serve transformed CSV for download (dtype: 'train' or 'test')"""
    if dtype == 'train':
        df = server_state.get('transformed_train')
        filename = 'transformed_train_data.csv'
    elif dtype == 'test':
        df = server_state.get('transformed_test')
        filename = 'transformed_test_data.csv'
    else:
        return jsonify({'error': 'Invalid type, use train or test'}), 400

    if df is None:
        return jsonify({'error': f'No transformed {dtype} data available'}), 404

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_bytes = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
    csv_bytes.seek(0)
    return send_file(csv_bytes, mimetype='text/csv', as_attachment=True, download_name=filename)

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
