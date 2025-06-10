const API_BASE_URL = 'http://localhost:8000';

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const predictBtn = document.getElementById('predictBtn');
const compareBtn = document.getElementById('compareBtn');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const errorMsg = document.getElementById('errorMsg');
const resultsSection = document.getElementById('resultsSection');
const sampleItems = document.querySelectorAll('.sample-item');

let selectedFile = null;
let selectedSamplePath = null;

// Disaster type icons
const disasterIcons = {
    'Water_Disaster': '🌊',
    'Land_Slide': '🏔️',
    'Urban_Fire': '🔥',
    'Earthquake': '🌋'
};

// Sample image click handlers
sampleItems.forEach(item => {
    item.addEventListener('click', () => {
        const disasterType = item.dataset.type;
        const imagePath = `samples/${disasterType}.jpg`;
        
        // Remove selection from other samples
        sampleItems.forEach(sample => sample.classList.remove('selected'));
        // Add selection to clicked sample
        item.classList.add('selected');
        
        // Set the sample image as preview
        loadSampleImage(imagePath, disasterType);
    });
});

// Upload area interactions
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
fileInput.addEventListener('change', handleFileSelect);

// Button interactions
predictBtn.addEventListener('click', () => makePrediction('tuned'));
compareBtn.addEventListener('click', compareModels);
clearBtn.addEventListener('click', clearAll);

function loadSampleImage(imagePath, disasterType) {
    selectedFile = null; // Clear file selection
    selectedSamplePath = imagePath;
    
    imagePreview.src = imagePath;
    imagePreview.onerror = function() {
        // If sample image fails to load, show placeholder
        const canvas = document.createElement('canvas');
        canvas.width = 400;
        canvas.height = 300;
        const ctx = canvas.getContext('2d');
        
        // Create gradient background
        const gradient = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
        gradient.addColorStop(0, '#667eea');
        gradient.addColorStop(1, '#764ba2');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Add icon and text
        ctx.fillStyle = 'white';
        ctx.font = 'bold 48px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(disasterIcons[disasterType], canvas.width/2, canvas.height/2 - 30);
        
        ctx.font = 'bold 24px Arial';
        ctx.fillText(disasterType.replace('_', ' '), canvas.width/2, canvas.height/2 + 30);
        
        ctx.font = '16px Arial';
        ctx.fillText('Sample Image', canvas.width/2, canvas.height/2 + 60);
        
        this.src = canvas.toDataURL();
    };
    
    previewSection.style.display = 'block';
    hideError();
    clearResults();
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file.');
        return;
    }

    selectedFile = file;
    selectedSamplePath = null; // Clear sample selection
    
    // Remove sample selection
    sampleItems.forEach(sample => sample.classList.remove('selected'));
    
    const reader = new FileReader();
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        previewSection.style.display = 'block';
        hideError();
        clearResults();
    };
    reader.readAsDataURL(file);
}

async function makePrediction(modelType = 'tuned') {
    if (!selectedFile && !selectedSamplePath) {
        showError('Please select an image or choose a sample first.');
        return;
    }

    showLoading();
    clearResults();

    const formData = new FormData();
    
    if (selectedFile) {
        formData.append('file', selectedFile);
    } else if (selectedSamplePath) {
        // For sample images, we need to fetch the image and convert it to a file
        try {
            const response = await fetch(selectedSamplePath);
            const blob = await response.blob();
            const file = new File([blob], selectedSamplePath.split('/').pop(), { type: blob.type });
            formData.append('file', file);
        } catch (error) {
            hideLoading();
            showError('Failed to load sample image. Please try uploading your own image.');
            return;
        }
    }
    
    formData.append('model_type', modelType);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (data.success) {
            displaySingleResult(data, modelType);
        } else {
            showError('Prediction failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Connection error. Make sure the API server is running.');
    } finally {
        hideLoading();
    }
}

async function compareModels() {
    if (!selectedFile && !selectedSamplePath) {
        showError('Please select an image or choose a sample first.');
        return;
    }

    showLoading();
    clearResults();

    const formData = new FormData();
    
    if (selectedFile) {
        formData.append('file', selectedFile);
    } else if (selectedSamplePath) {
        try {
            const response = await fetch(selectedSamplePath);
            const blob = await response.blob();
            const file = new File([blob], selectedSamplePath.split('/').pop(), { type: blob.type });
            formData.append('file', file);
        } catch (error) {
            hideLoading();
            showError('Failed to load sample image. Please try uploading your own image.');
            return;
        }
    }

    try {
        const response = await fetch(`${API_BASE_URL}/compare`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        
        if (data.success) {
            displayComparisonResults(data);
        } else {
            showError('Prediction failed. Please try again.');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Connection error. Make sure the API server is running.');
    } finally {
        hideLoading();
    }
}

function displaySingleResult(data, modelType) {
    const modelTitle = modelType === 'tuned' ? 'Fine-tuned Model' : 'Base Model';
    
    resultsSection.innerHTML = `
        <div class="result-card">
            <div class="result-title">
                <span class="disaster-icons">🤖</span>
                ${modelTitle} Prediction
            </div>
            <div class="prediction-main">
                <div class="prediction-class">
                    ${disasterIcons[data.predicted_class] || '🌪️'} ${data.predicted_class.replace('_', ' ')}
                </div>
                <div class="confidence">
                    Confidence: ${(data.confidence * 100).toFixed(1)}%
                </div>
            </div>
            <div class="all-predictions">
                ${Object.entries(data.all_predictions)
                    .sort(([,a], [,b]) => b - a)
                    .map(([label, prob]) => `
                        <div class="prediction-item">
                            <div class="prediction-label">
                                ${disasterIcons[label] || '🌪️'} ${label.replace('_', ' ')}
                            </div>
                            <div class="prediction-bar">
                                <div class="prediction-fill" style="width: ${prob * 100}%"></div>
                            </div>
                            <div class="prediction-value">${(prob * 100).toFixed(1)}%</div>
                        </div>
                    `).join('')}
            </div>
        </div>
    `;
    resultsSection.style.display = 'block';
}

function displayComparisonResults(data) {
    resultsSection.innerHTML = `
        <div class="compare-section">
            <div class="result-card">
                <div class="result-title">
                    <span class="disaster-icons">🤖</span>
                    Base Model
                </div>
                <div class="prediction-main">
                    <div class="prediction-class">
                        ${disasterIcons[data.before_tuning.predicted_class] || '🌪️'} ${data.before_tuning.predicted_class.replace('_', ' ')}
                    </div>
                    <div class="confidence">
                        Confidence: ${(data.before_tuning.confidence * 100).toFixed(1)}%
                    </div>
                </div>
                <div class="all-predictions">
                    ${Object.entries(data.before_tuning.all_predictions)
                        .sort(([,a], [,b]) => b - a)
                        .map(([label, prob]) => `
                            <div class="prediction-item">
                                <div class="prediction-label">
                                    ${disasterIcons[label] || '🌪️'} ${label.replace('_', ' ')}
                                </div>
                                <div class="prediction-bar">
                                    <div class="prediction-fill" style="width: ${prob * 100}%"></div>
                                </div>
                                <div class="prediction-value">${(prob * 100).toFixed(1)}%</div>
                            </div>
                        `).join('')}
                </div>
            </div>
            
            <div class="result-card">
                <div class="result-title">
                    <span class="disaster-icons">🚀</span>
                    Fine-tuned Model
                </div>
                <div class="prediction-main">
                    <div class="prediction-class">
                        ${disasterIcons[data.after_tuning.predicted_class] || '🌪️'} ${data.after_tuning.predicted_class.replace('_', ' ')}
                    </div>
                    <div class="confidence">
                        Confidence: ${(data.after_tuning.confidence * 100).toFixed(1)}%
                    </div>
                </div>
                <div class="all-predictions">
                    ${Object.entries(data.after_tuning.all_predictions)
                        .sort(([,a], [,b]) => b - a)
                        .map(([label, prob]) => `
                            <div class="prediction-item">
                                <div class="prediction-label">
                                    ${disasterIcons[label] || '🌪️'} ${label.replace('_', ' ')}
                                </div>
                                <div class="prediction-bar">
                                    <div class="prediction-fill" style="width: ${prob * 100}%"></div>
                                </div>
                                <div class="prediction-value">${(prob * 100).toFixed(1)}%</div>
                            </div>
                        `).join('')}
                </div>
            </div>
        </div>
    `;
    resultsSection.style.display = 'block';
}

function showLoading() {
    loading.style.display = 'block';
    predictBtn.disabled = true;
    compareBtn.disabled = true;
}

function hideLoading() {
    loading.style.display = 'none';
    predictBtn.disabled = false;
    compareBtn.disabled = false;
}

function showError(message) {
    errorMsg.textContent = message;
    errorMsg.style.display = 'block';
}

function hideError() {
    errorMsg.style.display = 'none';
}

function clearResults() {
    resultsSection.style.display = 'none';
    resultsSection.innerHTML = '';
}

function clearAll() {
    selectedFile = null;
    selectedSamplePath = null;
    previewSection.style.display = 'none';
    imagePreview.src = '';
    fileInput.value = '';
    sampleItems.forEach(sample => sample.classList.remove('selected'));
    clearResults();
    hideError();
}