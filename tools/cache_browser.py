#!/usr/bin/env python3
"""
NAD Cache Browser - Web interface for browsing available caches

Usage:
    python3 tools/cache_browser.py [--port PORT] [--host HOST]

Features:
    - List all MUI_HUB caches organized by model and dataset
    - List local cache_* directories
    - Show cache metadata (samples, problems, version, etc.)
    - Quick links to visualization server
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, jsonify, redirect, url_for

app = Flask(__name__)

# Configuration
BASE_DIR = Path(__file__).parent.parent
MUI_PUBLIC_DIR = BASE_DIR / "MUI_HUB" / "cache"
VISUALIZATION_PORT = 5002

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NAD Cache Browser</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
            min-height: 100vh;
            color: #333;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            padding: 30px 0;
            margin-bottom: 30px;
        }

        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #2563eb, #059669);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #666;
            font-size: 1.1rem;
        }

        .stats-bar {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 20px 0;
            flex-wrap: wrap;
        }

        .stat-item {
            text-align: center;
            padding: 15px 25px;
            background: #fff;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }

        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2563eb;
        }

        .stat-label {
            font-size: 0.85rem;
            color: #666;
            margin-top: 5px;
        }

        .section {
            margin-bottom: 40px;
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #2563eb;
        }

        .section-title {
            font-size: 1.5rem;
            color: #2563eb;
        }

        .section-count {
            background: #2563eb;
            color: #fff;
            padding: 2px 10px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
        }

        .model-card {
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 15px;
            margin-bottom: 20px;
            overflow: hidden;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }

        .model-card:hover {
            border-color: #2563eb;
            box-shadow: 0 4px 20px rgba(37, 99, 235, 0.15);
        }

        .model-header {
            background: linear-gradient(90deg, #eff6ff, #f0fdf4);
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .model-name {
            font-size: 1.2rem;
            font-weight: 600;
            color: #059669;
        }

        .model-meta {
            display: flex;
            gap: 15px;
            font-size: 0.85rem;
            color: #666;
        }

        .cache-card {
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }

        .cache-card:hover {
            background: #fff;
            border-color: #059669;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(5, 150, 105, 0.15);
        }

        .cache-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 10px;
        }

        .dataset-badge {
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 0.75rem;
            font-weight: bold;
        }

        .badge-math { background: #ef4444; color: #fff; }
        .badge-code { background: #06b6d4; color: #fff; }
        .badge-science { background: #f59e0b; color: #fff; }
        .badge-other { background: #6b7280; color: #fff; }

        .cache-badges {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }

        .cache-type-badge {
            padding: 3px 8px;
            border-radius: 5px;
            font-size: 0.75rem;
            font-weight: bold;
            background: #e0e7ff;
            color: #3730a3;
        }

        .datasets-container {
            padding: 15px;
            background: #fafafa;
        }

        .dataset-card {
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            margin-bottom: 15px;
            overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }

        .dataset-card:last-child {
            margin-bottom: 0;
        }

        .dataset-header {
            background: linear-gradient(90deg, #f8fafc, #f1f5f9);
            padding: 12px 15px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #e0e0e0;
            transition: background 0.2s ease;
        }

        .dataset-header:hover {
            background: linear-gradient(90deg, #f1f5f9, #e2e8f0);
        }

        .dataset-name {
            font-weight: 600;
            color: #1e40af;
            font-size: 1rem;
        }

        .dataset-meta {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.85rem;
        }

        .cache-count {
            color: #666;
        }

        .toggle-icon {
            color: #666;
            transition: transform 0.2s ease;
        }

        .cache-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            padding: 15px;
            background: #f8fafc;
        }

        .cache-info {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
            margin: 10px 0;
            font-size: 0.85rem;
        }

        .cache-info-item {
            display: flex;
            gap: 5px;
        }

        .info-label {
            color: #666;
        }

        .info-value {
            color: #2563eb;
            font-weight: 500;
        }

        .accuracy-high {
            color: #059669;
            font-weight: 600;
        }

        .accuracy-mid {
            color: #d97706;
            font-weight: 600;
        }

        .accuracy-low {
            color: #dc2626;
            font-weight: 600;
        }

        /* Modal styles */
        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }

        .modal.show {
            display: flex;
        }

        .modal-content {
            background: #fff;
            border-radius: 12px;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            overflow: hidden;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            animation: modalSlide 0.2s ease;
        }

        @keyframes modalSlide {
            from {
                transform: translateY(-20px);
                opacity: 0;
            }
            to {
                transform: translateY(0);
                opacity: 1;
            }
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px 20px;
            border-bottom: 1px solid #e5e7eb;
            background: #f8fafc;
        }

        .modal-title {
            font-size: 1.1rem;
            color: #1e40af;
            margin: 0;
        }

        .modal-close {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: #666;
            padding: 0 5px;
            line-height: 1;
        }

        .modal-close:hover {
            color: #dc2626;
        }

        .modal-copy-close {
            background: linear-gradient(90deg, #2563eb, #059669);
            color: #fff;
            border: none;
            padding: 6px 12px;
            border-radius: 5px;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .modal-copy-close:hover {
            transform: scale(1.05);
            box-shadow: 0 2px 8px rgba(37, 99, 235, 0.4);
        }

        .modal-copy-close.copied {
            background: linear-gradient(90deg, #059669, #10b981);
        }

        .modal-body {
            padding: 20px;
            overflow-y: auto;
            max-height: calc(80vh - 60px);
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }

        .detail-section {
            margin-bottom: 20px;
        }

        .detail-section:last-child {
            margin-bottom: 0;
        }

        .detail-section-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #1e40af;
            margin-bottom: 10px;
            padding-bottom: 5px;
            border-bottom: 1px solid #e5e7eb;
        }

        .detail-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
        }

        .detail-item {
            display: flex;
            flex-direction: column;
        }

        .detail-label {
            font-size: 0.75rem;
            color: #666;
            margin-bottom: 2px;
        }

        .detail-value {
            font-size: 0.9rem;
            color: #333;
            font-weight: 500;
        }

        .detail-files {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.8rem;
            background: #f3f4f6;
            padding: 10px;
            border-radius: 5px;
            max-height: 150px;
            overflow-y: auto;
        }

        .detail-file {
            padding: 3px 0;
            color: #374151;
        }

        .detail-file-dir {
            color: #2563eb;
        }

        .cache-path {
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.75rem;
            color: #666;
            background: #f3f4f6;
            padding: 8px;
            border-radius: 5px;
            word-break: break-all;
            margin-top: 10px;
            border: 1px solid #e5e7eb;
        }

        .cache-actions {
            display: flex;
            gap: 10px;
            margin-top: 12px;
        }

        .btn {
            padding: 8px 15px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 500;
            transition: all 0.2s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 5px;
        }

        .btn-primary {
            background: linear-gradient(90deg, #2563eb, #059669);
            color: #fff;
        }

        .btn-primary:hover {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
        }

        .btn-secondary {
            background: #f3f4f6;
            color: #333;
            border: 1px solid #e0e0e0;
        }

        .btn-secondary:hover {
            background: #e5e7eb;
            border-color: #d1d5db;
        }

        .btn-copy {
            background: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
        }

        .btn-copy:hover {
            background: #fde68a;
            border-color: #f59e0b;
        }

        .btn-copy.copied {
            background: #d1fae5;
            color: #065f46;
            border-color: #10b981;
        }

        .local-cache-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }

        .local-cache-card {
            background: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
        }

        .local-cache-card:hover {
            background: #fff;
            border-color: #f59e0b;
            box-shadow: 0 4px 12px rgba(245, 158, 11, 0.15);
        }

        .local-cache-name {
            font-weight: 600;
            color: #d97706;
            font-size: 1.1rem;
            margin-bottom: 10px;
        }

        .empty-state {
            text-align: center;
            padding: 40px;
            color: #666;
            background: #fff;
            border-radius: 10px;
            border: 1px dashed #d1d5db;
        }

        .refresh-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(90deg, #2563eb, #059669);
            border: none;
            cursor: pointer;
            font-size: 1.5rem;
            color: #fff;
            box-shadow: 0 4px 20px rgba(37, 99, 235, 0.4);
            transition: all 0.3s ease;
        }

        .refresh-btn:hover {
            transform: scale(1.1) rotate(180deg);
        }

        .search-box {
            width: 100%;
            max-width: 500px;
            margin: 0 auto 30px;
            position: relative;
        }

        .search-input {
            width: 100%;
            padding: 15px 20px;
            padding-left: 50px;
            border: 2px solid #e0e0e0;
            border-radius: 30px;
            background: #fff;
            color: #333;
            font-size: 1rem;
            outline: none;
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }

        .search-input:focus {
            border-color: #2563eb;
            box-shadow: 0 0 20px rgba(37, 99, 235, 0.2);
        }

        .search-icon {
            position: absolute;
            left: 20px;
            top: 50%;
            transform: translateY(-50%);
            color: #666;
        }

        @media (max-width: 768px) {
            .dataset-grid {
                grid-template-columns: 1fr;
            }
            .stats-bar {
                gap: 15px;
            }
            .stat-item {
                padding: 10px 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>NAD Cache Browser</h1>
            <p class="subtitle">Neuron Activation Distribution Cache Management</p>
        </header>

        <div class="stats-bar">
            <div class="stat-item">
                <div class="stat-value">{{ stats.total_models }}</div>
                <div class="stat-label">Models</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ stats.total_datasets }}</div>
                <div class="stat-label">Datasets</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ stats.total_caches }}</div>
                <div class="stat-label">Total Caches</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{{ stats.total_samples }}</div>
                <div class="stat-label">Total Samples</div>
            </div>
        </div>

        <div class="search-box">
            <span class="search-icon">&#128269;</span>
            <input type="text" class="search-input" placeholder="Search models, datasets, or caches..." id="searchInput" onkeyup="filterCaches()">
        </div>

        <!-- MUI Public Caches -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">MUI Public Caches</h2>
                <span class="section-count">{{ mui_caches|length }} models</span>
            </div>

            {% if mui_caches %}
                {% for model_name, datasets in mui_caches.items() %}
                <div class="model-card" data-model="{{ model_name }}">
                    <div class="model-header" onclick="toggleModel(this)">
                        <span class="model-name">{{ model_name }}</span>
                        <span class="model-meta">
                            <span>{{ datasets|length }} datasets</span>
                            <span>&#9660;</span>
                        </span>
                    </div>
                    <div class="datasets-container">
                        {% for dataset_name, cache_list in datasets.items() %}
                        <div class="dataset-card" data-dataset="{{ dataset_name }}">
                            <div class="dataset-header" onclick="toggleDataset(this)">
                                <span class="dataset-name">{{ dataset_name }}</span>
                                <span class="dataset-meta">
                                    <span class="dataset-badge badge-{{ cache_list[0].category }}">{{ cache_list[0].category }}</span>
                                    <span class="cache-count">{{ cache_list|length }} caches</span>
                                    <span class="toggle-icon">&#9660;</span>
                                </span>
                            </div>
                            <div class="cache-grid">
                                {% for cache_info in cache_list %}
                                <div class="cache-card">
                                    <div class="cache-header">
                                        <span class="cache-type-badge">{{ cache_info.cache_type }}</span>
                                    </div>
                                    <div class="cache-info">
                                        <div class="cache-info-item">
                                            <span class="info-label">Samples:</span>
                                            <span class="info-value">{{ cache_info.num_samples or 'N/A' }}</span>
                                        </div>
                                        <div class="cache-info-item">
                                            <span class="info-label">Problems:</span>
                                            <span class="info-value">{{ cache_info.num_problems or 'N/A' }}</span>
                                        </div>
                                        <div class="cache-info-item">
                                            <span class="info-label">Temp:</span>
                                            <span class="info-value">{{ cache_info.temperature if cache_info.temperature is not none else 'N/A' }}</span>
                                        </div>
                                        <div class="cache-info-item">
                                            <span class="info-label">Accuracy:</span>
                                            <span class="info-value {% if cache_info.accuracy is not none %}{% if cache_info.accuracy >= 80 %}accuracy-high{% elif cache_info.accuracy >= 50 %}accuracy-mid{% else %}accuracy-low{% endif %}{% endif %}">
                                                {{ (cache_info.accuracy|string + '%') if cache_info.accuracy is not none else 'N/A' }}
                                            </span>
                                        </div>
                                        <div class="cache-info-item" style="white-space:nowrap">
                                            <span class="info-label">Date:</span>
                                            <span class="info-value">{{ cache_info.date or 'N/A' }}</span>
                                        </div>
                                    </div>
                                    <div class="cache-path">{{ cache_info.path }}</div>
                                    <div class="cache-actions">
                                        <button class="btn btn-primary" onclick="showDetails('{{ cache_info.path }}')">
                                            &#128196; Details
                                        </button>
                                        <button class="btn btn-copy" onclick="copyPath('{{ cache_info.abs_path }}', this)">
                                            &#128203; Copy
                                        </button>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endfor %}
            {% else %}
                <div class="empty-state">
                    <p>No MUI HUB caches found in MUI_HUB/cache/</p>
                </div>
            {% endif %}
        </div>

        <!-- Local Caches -->
        <div class="section">
            <div class="section-header">
                <h2 class="section-title">Local Caches</h2>
                <span class="section-count">{{ local_caches|length }} caches</span>
            </div>

            {% if local_caches %}
            <div class="local-cache-grid">
                {% for cache in local_caches %}
                <div class="local-cache-card" data-cache="{{ cache.name }}">
                    <div class="local-cache-name">{{ cache.name }}</div>
                    <div class="cache-info">
                        <div class="cache-info-item">
                            <span class="info-label">Samples:</span>
                            <span class="info-value">{{ cache.num_samples or 'N/A' }}</span>
                        </div>
                        <div class="cache-info-item">
                            <span class="info-label">Problems:</span>
                            <span class="info-value">{{ cache.num_problems or 'N/A' }}</span>
                        </div>
                        <div class="cache-info-item">
                            <span class="info-label">Version:</span>
                            <span class="info-value">{{ cache.version or 'N/A' }}</span>
                        </div>
                        <div class="cache-info-item">
                            <span class="info-label">Size:</span>
                            <span class="info-value">{{ cache.size or 'N/A' }}</span>
                        </div>
                    </div>
                    <div class="cache-path">{{ cache.path }}</div>
                    <div class="cache-actions">
                        <button class="btn btn-primary" onclick="showDetails('{{ cache.path }}')">
                            &#128196; Details
                        </button>
                        <button class="btn btn-copy" onclick="copyPath('{{ cache.abs_path }}', this)">
                            &#128203; Copy
                        </button>
                    </div>
                </div>
                {% endfor %}
            </div>
            {% else %}
                <div class="empty-state">
                    <p>No local caches found (cache_* directories)</p>
                </div>
            {% endif %}
        </div>
    </div>

    <button class="refresh-btn" onclick="location.reload()" title="Refresh">&#8635;</button>

    <!-- Details Modal -->
    <div id="detailsModal" class="modal" onclick="closeModalOnBackground(event)">
        <div class="modal-content">
            <div class="modal-header">
                <button class="modal-copy-close" onclick="copyAndClose()" title="Copy path and close">
                    &#128203; Copy & Close
                </button>
                <h3 class="modal-title">Cache Details</h3>
                <button class="modal-close" onclick="closeModal()">&times;</button>
            </div>
            <div class="modal-body" id="modalBody">
                <div class="loading">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        let currentCachePath = '';

        function showDetails(cachePath) {
            const modal = document.getElementById('detailsModal');
            const modalBody = document.getElementById('modalBody');

            // Show modal with loading
            modal.classList.add('show');
            modalBody.innerHTML = '<div class="loading">Loading...</div>';

            // Fetch details
            fetch('/api/cache/' + cachePath)
                .then(response => response.json())
                .then(data => {
                    // Store the absolute path from API response for copying
                    currentCachePath = data.abs_path || data.path || cachePath;
                    let html = '';

                    // Model info section
                    if (data.model || data.model_path || data.report_path) {
                        html += '<div class="detail-section">';
                        html += '<div class="detail-section-title">Model Information</div>';
                        html += '<div class="detail-grid" style="grid-template-columns: 1fr;">';
                        if (data.model) {
                            html += `<div class="detail-item"><span class="detail-label">Model</span><span class="detail-value" style="word-break:break-all">${data.model}</span></div>`;
                        }
                        if (data.model_path) {
                            html += `<div class="detail-item"><span class="detail-label">Model Path</span><span class="detail-value" style="word-break:break-all">${data.model_path}</span></div>`;
                        }
                        if (data.report_path) {
                            html += `<div class="detail-item"><span class="detail-label">Report Path</span><span class="detail-value" style="word-break:break-all">${data.report_path}</span></div>`;
                        }
                        html += '</div></div>';
                    }

                    // Basic info section
                    html += '<div class="detail-section">';
                    html += '<div class="detail-section-title">Basic Information</div>';
                    html += '<div class="detail-grid">';
                    html += `<div class="detail-item"><span class="detail-label">Samples</span><span class="detail-value">${data.num_samples || 'N/A'}</span></div>`;
                    html += `<div class="detail-item"><span class="detail-label">Problems</span><span class="detail-value">${data.num_problems || 'N/A'}</span></div>`;
                    html += `<div class="detail-item"><span class="detail-label">Accuracy</span><span class="detail-value">${data.accuracy ? data.accuracy + '%' : 'N/A'}</span></div>`;
                    html += `<div class="detail-item"><span class="detail-label">Date</span><span class="detail-value">${data.date || 'N/A'}</span></div>`;
                    html += `<div class="detail-item"><span class="detail-label">Version</span><span class="detail-value">${data.version || 'N/A'}</span></div>`;
                    html += `<div class="detail-item"><span class="detail-label">Size</span><span class="detail-value">${data.size || 'N/A'}</span></div>`;
                    html += '</div></div>';

                    // Sampling params section
                    if (data.sampling_params) {
                        html += '<div class="detail-section">';
                        html += '<div class="detail-section-title">Sampling Parameters</div>';
                        html += '<div class="detail-grid">';
                        const sp = data.sampling_params;
                        html += `<div class="detail-item"><span class="detail-label">Temperature</span><span class="detail-value">${sp.temperature ?? 'N/A'}</span></div>`;
                        html += `<div class="detail-item"><span class="detail-label">Top-p</span><span class="detail-value">${sp.top_p ?? 'N/A'}</span></div>`;
                        html += `<div class="detail-item"><span class="detail-label">Top-k</span><span class="detail-value">${sp.top_k ?? 'N/A'}</span></div>`;
                        html += `<div class="detail-item"><span class="detail-label">Max Tokens</span><span class="detail-value">${sp.max_tokens ?? 'N/A'}</span></div>`;
                        if (sp.prompt_mode) {
                            html += `<div class="detail-item"><span class="detail-label">Prompt Mode</span><span class="detail-value">${sp.prompt_mode}</span></div>`;
                        }
                        html += '</div></div>';
                    }

                    // Files section
                    if (data.files && data.files.length > 0) {
                        html += '<div class="detail-section">';
                        html += '<div class="detail-section-title">Files</div>';
                        html += '<div class="detail-files">';
                        data.files.forEach(f => {
                            const isDir = f.type === 'directory';
                            html += `<div class="detail-file ${isDir ? 'detail-file-dir' : ''}">${f.name}${f.size ? ' (' + f.size + ')' : ''}</div>`;
                        });
                        html += '</div></div>';
                    }

                    // Path section
                    html += '<div class="detail-section">';
                    html += '<div class="detail-section-title">Path</div>';
                    html += `<div class="cache-path" style="margin:0">${data.abs_path || data.path}</div>`;
                    html += '</div>';

                    modalBody.innerHTML = html;
                })
                .catch(err => {
                    modalBody.innerHTML = '<div class="loading" style="color:#dc2626">Error loading details</div>';
                });
        }

        function closeModal() {
            document.getElementById('detailsModal').classList.remove('show');
        }

        function copyAndClose() {
            const btn = document.querySelector('.modal-copy-close');
            if (!currentCachePath) {
                alert('No path to copy');
                return;
            }

            // Use fallback method for better compatibility
            const textArea = document.createElement('textarea');
            textArea.value = currentCachePath;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '0';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();

            try {
                document.execCommand('copy');
                btn.innerHTML = '&#10003; Copied!';
                btn.classList.add('copied');
                setTimeout(function() {
                    closeModal();
                    btn.innerHTML = '&#128203; Copy & Close';
                    btn.classList.remove('copied');
                }, 500);
            } catch (err) {
                alert('Copy failed: ' + err);
            }

            document.body.removeChild(textArea);
        }

        function closeModalOnBackground(event) {
            if (event.target.classList.contains('modal')) {
                closeModal();
            }
        }

        // Close modal on Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') closeModal();
        });

        function copyPath(path, btn) {
            const originalText = btn.innerHTML;

            // Use textarea fallback for better compatibility (non-HTTPS)
            const textArea = document.createElement('textarea');
            textArea.value = path;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '0';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();

            let success = false;
            try {
                success = document.execCommand('copy');
            } catch (e) {
                success = false;
            }
            document.body.removeChild(textArea);

            if (success) {
                btn.innerHTML = '&#10003; Copied!';
                btn.classList.add('copied');
                setTimeout(function() {
                    btn.innerHTML = originalText;
                    btn.classList.remove('copied');
                }, 2000);
            } else {
                // Try navigator.clipboard as fallback
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(path).then(function() {
                        btn.innerHTML = '&#10003; Copied!';
                        btn.classList.add('copied');
                        setTimeout(function() {
                            btn.innerHTML = originalText;
                            btn.classList.remove('copied');
                        }, 2000);
                    }).catch(function() {
                        alert('Copy failed. Path: ' + path);
                    });
                } else {
                    alert('Copy failed. Path: ' + path);
                }
            }
        }

        function toggleModel(element) {
            const container = element.parentElement.querySelector('.datasets-container');
            if (container.style.display === 'none') {
                container.style.display = 'block';
                element.querySelector('.model-meta span:last-child').innerHTML = '&#9660;';
            } else {
                container.style.display = 'none';
                element.querySelector('.model-meta span:last-child').innerHTML = '&#9654;';
            }
        }

        function toggleDataset(element) {
            const grid = element.parentElement.querySelector('.cache-grid');
            const icon = element.querySelector('.toggle-icon');
            if (grid.style.display === 'none') {
                grid.style.display = 'grid';
                icon.innerHTML = '&#9660;';
                icon.style.transform = 'rotate(0deg)';
            } else {
                grid.style.display = 'none';
                icon.innerHTML = '&#9654;';
                icon.style.transform = 'rotate(0deg)';
            }
        }

        function filterCaches() {
            const query = document.getElementById('searchInput').value.toLowerCase();

            // Filter model cards
            document.querySelectorAll('.model-card').forEach(modelCard => {
                const modelName = modelCard.dataset.model.toLowerCase();
                let hasVisibleDataset = false;

                // Filter dataset cards within this model
                modelCard.querySelectorAll('.dataset-card').forEach(datasetCard => {
                    const datasetName = datasetCard.dataset.dataset.toLowerCase();
                    let hasVisibleCache = false;

                    // Filter cache cards within this dataset
                    datasetCard.querySelectorAll('.cache-card').forEach(cacheCard => {
                        const path = cacheCard.querySelector('.cache-path').textContent.toLowerCase();
                        const cacheType = cacheCard.querySelector('.cache-type-badge')?.textContent.toLowerCase() || '';

                        if (modelName.includes(query) || datasetName.includes(query) ||
                            path.includes(query) || cacheType.includes(query)) {
                            cacheCard.style.display = 'block';
                            hasVisibleCache = true;
                        } else {
                            cacheCard.style.display = 'none';
                        }
                    });

                    datasetCard.style.display = hasVisibleCache ? 'block' : 'none';
                    if (hasVisibleCache) hasVisibleDataset = true;

                    // Expand dataset if searching and has matches
                    if (query && hasVisibleCache) {
                        datasetCard.querySelector('.cache-grid').style.display = 'grid';
                    }
                });

                modelCard.style.display = hasVisibleDataset || modelName.includes(query) ? 'block' : 'none';

                // Expand model if searching and has matches
                if (query && hasVisibleDataset) {
                    modelCard.querySelector('.datasets-container').style.display = 'block';
                }
            });

            // Filter local caches
            document.querySelectorAll('.local-cache-card').forEach(card => {
                const cacheName = card.dataset.cache.toLowerCase();
                const path = card.querySelector('.cache-path').textContent.toLowerCase();

                if (cacheName.includes(query) || path.includes(query)) {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            });
        }
    </script>
</body>
</html>
"""


def get_dataset_category(dataset_name: str) -> str:
    """Determine dataset category based on name."""
    name = dataset_name.lower()
    if any(x in name for x in ['aime', 'math', 'gsm', 'minerva']):
        return 'math'
    elif any(x in name for x in ['humaneval', 'mbpp', 'livecodebench', 'code']):
        return 'code'
    elif any(x in name for x in ['gpqa', 'science', 'arc']):
        return 'science'
    return 'other'


def get_cache_metadata(cache_path: Path) -> dict:
    """Extract metadata from a cache directory."""
    info = {
        'path': str(cache_path),
        'num_samples': None,
        'num_problems': None,
        'version': None,
        'date': None,
        'temperature': None,
        'top_p': None,
        'accuracy': None,
        'correct_runs': None,
        'total_runs': None,
        'report_path': None,
        'model_path': None,
        'model': None,
    }

    # Try to read meta.json
    meta_file = cache_path / 'meta.json'
    if meta_file.exists():
        try:
            with open(meta_file) as f:
                meta = json.load(f)
                if 'samples' in meta:
                    info['num_samples'] = len(meta['samples'])
                    problems = set(s.get('problem_id') for s in meta['samples'] if s.get('problem_id'))
                    info['num_problems'] = len(problems)
                # Get report_path and model_path
                info['report_path'] = meta.get('report_path')
                info['model_path'] = meta.get('model_path')
        except Exception:
            pass

    # Try to read evaluation_report_compact.json for sampling params and accuracy
    eval_file = cache_path / 'evaluation_report_compact.json'
    if eval_file.exists():
        try:
            with open(eval_file) as f:
                eval_data = json.load(f)
                sampling_params = eval_data.get('sampling_params', {})
                info['temperature'] = sampling_params.get('temperature')
                info['top_p'] = sampling_params.get('top_p')

                # Get model from test_info
                test_info = eval_data.get('test_info', {})
                info['model'] = test_info.get('model')

                # Calculate accuracy from runs' is_correct
                total_runs = 0
                correct_runs = 0
                results = eval_data.get('results', [])
                for result in results:
                    for run in result.get('runs', []):
                        total_runs += 1
                        if run.get('is_correct'):
                            correct_runs += 1

                if total_runs > 0:
                    info['accuracy'] = round(correct_runs / total_runs * 100, 2)
                    info['correct_runs'] = correct_runs
                    info['total_runs'] = total_runs
        except Exception:
            pass

    # Try to read manifest.json
    manifest_file = cache_path / 'manifest.json'
    if manifest_file.exists():
        try:
            with open(manifest_file) as f:
                manifest = json.load(f)
                info['version'] = manifest.get('schema_version', manifest.get('version'))
        except Exception:
            pass

    # Extract date from directory name
    dir_name = cache_path.name
    # Pattern: cache_neuron_output_1_act_no_rms_YYYYMMDD_HHMMSS
    import re
    date_match = re.search(r'(\d{8})_(\d{6})$', dir_name)
    if date_match:
        date_str = date_match.group(1)
        info['date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

    return info


def get_directory_size(path: Path) -> str:
    """Get human-readable directory size."""
    try:
        total = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total < 1024:
                return f"{total:.1f} {unit}"
            total /= 1024
        return f"{total:.1f} TB"
    except Exception:
        return 'N/A'


def get_cache_type(cache_name: str) -> str:
    """Extract cache type from cache directory name."""
    # Pattern: cache_neuron_output_X_TYPE_...
    # Examples: cache_neuron_output_1_act_no_rms_..., cache_neuron_output_2_down_pre_no_rms_...
    import re
    match = re.search(r'cache_neuron_output_(\d+)_([a-z_]+?)_(?:no_rms_)?\d{8}', cache_name)
    if match:
        layer_num = match.group(1)
        cache_type = match.group(2).replace('_', ' ').title()
        return f"L{layer_num} {cache_type}"
    return cache_name[:30]


def scan_mui_caches() -> dict:
    """Scan MUI_HUB/cache directory for caches."""
    caches = {}

    if not MUI_PUBLIC_DIR.exists():
        return caches

    for model_dir in sorted(MUI_PUBLIC_DIR.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        caches[model_name] = {}

        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name
            cache_list = []

            # Find ALL cache directories
            for cache_dir in sorted(dataset_dir.iterdir()):
                if cache_dir.is_dir() and cache_dir.name.startswith('cache_neuron_output'):
                    cache_info = get_cache_metadata(cache_dir)
                    cache_info['category'] = get_dataset_category(dataset_name)
                    cache_info['cache_type'] = get_cache_type(cache_dir.name)
                    cache_info['cache_name'] = cache_dir.name
                    # Use relative path for API compatibility
                    try:
                        cache_info['path'] = str(cache_dir.relative_to(BASE_DIR))
                    except ValueError:
                        cache_info['path'] = str(cache_dir)
                    # Absolute path for copying
                    cache_info['abs_path'] = str(cache_dir.resolve())
                    cache_list.append(cache_info)

            if cache_list:
                caches[model_name][dataset_name] = cache_list

    # Remove empty models
    caches = {k: v for k, v in caches.items() if v}

    return caches


def scan_local_caches() -> list:
    """Scan local cache_* directories."""
    caches = []

    for item in sorted(BASE_DIR.iterdir()):
        if item.is_dir() and item.name.startswith('cache_'):
            cache_info = get_cache_metadata(item)
            cache_info['name'] = item.name
            cache_info['size'] = get_directory_size(item)
            # Use relative path
            cache_info['path'] = item.name
            # Absolute path for copying
            cache_info['abs_path'] = str(item.resolve())
            caches.append(cache_info)

    return caches


def calculate_stats(mui_caches: dict, local_caches: list) -> dict:
    """Calculate summary statistics."""
    total_samples = 0
    total_datasets = 0
    total_caches = 0

    for model, datasets in mui_caches.items():
        total_datasets += len(datasets)
        for dataset, cache_list in datasets.items():
            total_caches += len(cache_list)
            for cache_info in cache_list:
                if cache_info.get('num_samples'):
                    total_samples += cache_info['num_samples']

    for cache in local_caches:
        if cache.get('num_samples'):
            total_samples += cache['num_samples']

    return {
        'total_models': len(mui_caches),
        'total_datasets': total_datasets,
        'total_caches': total_caches + len(local_caches),
        'total_samples': f"{total_samples:,}" if total_samples else 'N/A',
    }


@app.route('/')
def index():
    """Main page showing all caches."""
    mui_caches = scan_mui_caches()
    local_caches = scan_local_caches()
    stats = calculate_stats(mui_caches, local_caches)

    return render_template_string(
        HTML_TEMPLATE,
        mui_caches=mui_caches,
        local_caches=local_caches,
        stats=stats,
    )


@app.route('/api/caches')
def api_caches():
    """API endpoint returning all caches as JSON."""
    mui_caches = scan_mui_caches()
    local_caches = scan_local_caches()

    return jsonify({
        'mui_public': mui_caches,
        'local': local_caches,
        'stats': calculate_stats(mui_caches, local_caches),
    })


@app.route('/api/cache/<path:cache_path>')
def api_cache_detail(cache_path):
    """API endpoint for single cache details."""
    full_path = BASE_DIR / cache_path

    if not full_path.exists():
        return jsonify({'error': 'Cache not found'}), 404

    info = get_cache_metadata(full_path)
    info['size'] = get_directory_size(full_path)
    info['abs_path'] = str(full_path.resolve())

    # Read full sampling_params from evaluation_report_compact.json
    eval_file = full_path / 'evaluation_report_compact.json'
    if eval_file.exists():
        try:
            with open(eval_file) as f:
                eval_data = json.load(f)
                info['sampling_params'] = eval_data.get('sampling_params', {})
                info['model_name'] = eval_data.get('model_name')
                info['total_problems'] = eval_data.get('total_problems')
                info['correct_count'] = eval_data.get('correct_count')
                info['accuracy'] = eval_data.get('accuracy')
        except Exception:
            pass

    # List files in cache
    files = []
    for item in full_path.iterdir():
        if item.is_file():
            files.append({
                'name': item.name,
                'size': f"{item.stat().st_size:,} bytes"
            })
        elif item.is_dir():
            files.append({
                'name': item.name + '/',
                'type': 'directory'
            })

    info['files'] = sorted(files, key=lambda x: x['name'])

    return jsonify(info)


def main():
    parser = argparse.ArgumentParser(description='NAD Cache Browser Web Server')
    parser.add_argument('--port', type=int, default=5003, help='Server port (default: 5003)')
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--vis-port', type=int, default=5002,
                        help='Visualization server port (default: 5002)')

    args = parser.parse_args()

    global VISUALIZATION_PORT
    VISUALIZATION_PORT = args.vis_port

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    NAD Cache Browser                         ║
╠══════════════════════════════════════════════════════════════╣
║  Server running at: http://{args.host}:{args.port:<24}  ║
║  Visualization port: {args.vis_port:<38} ║
╚══════════════════════════════════════════════════════════════╝
    """)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    from flask import request
    main()
