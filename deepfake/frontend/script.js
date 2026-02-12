const API_URL = 'http://localhost:3000/api';

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadBox = document.getElementById('uploadBox');
const uploadSection = document.getElementById('uploadSection');
const previewSection = document.getElementById('previewSection');
const previewImg = document.getElementById('previewImg');
const loadingSection = document.getElementById('loadingSection');
const loadingStatus = document.getElementById('loadingStatus');
const progressBar = document.getElementById('progressBar');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultSection = document.getElementById('resultSection');
const statsPanel = document.getElementById('statsPanel');

// Result elements
const resultIcon = document.getElementById('resultIcon');
const resultLabel = document.getElementById('resultLabel');
const resultSubtitle = document.getElementById('resultSubtitle');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const processingTime = document.getElementById('processingTime');
const dataSource = document.getElementById('dataSource');
const timestamp = document.getElementById('timestamp');

// Stats elements
const totalScans = document.getElementById('totalScans');
const realCount = document.getElementById('realCount');
const fakeCount = document.getElementById('fakeCount');
const avgConfidence = document.getElementById('avgConfidence');

let selectedFile = null;


// FILE UPLOAD HANDLING

fileInput.addEventListener('change', handleFileSelect);

uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        fileInput.files = files;
        handleFileSelect();
    }
});

function handleFileSelect() {
    const file = fileInput.files[0];
    
    if (!file) return;
    
    // Validate file type
    if (!file.type.match('image.*')) {
        showToast('Please select an image file', 'error');
        return;
    }
    
    // Validate file size (10MB max)
    if (file.size > 10 * 1024 * 1024) {
        showToast('Image size must be less than 10MB', 'error');
        return;
    }
    
    selectedFile = file;
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        uploadSection.style.display = 'none';
        previewSection.style.display = 'block';
        analyzeBtn.style.display = 'flex';
        resultSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// ANALYZE IMAGE

async function analyzeImage() {
    if (!selectedFile) {
        showToast('Please select an image first', 'error');
        return;
    }
    
    // Hide preview and analyze button
    previewSection.style.display = 'none';
    analyzeBtn.style.display = 'none';
    resultSection.style.display = 'none';
    
    // Show loading
    loadingSection.style.display = 'block';
    simulateProgress();
    
    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);
    
    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showResult(data);
            updateStats();
        } else {
            showToast('Analysis failed: ' + data.error, 'error');
            resetUpload();
        }
        
    } catch (error) {
        console.error('Error:', error);
        showToast('Network error. Please check if server is running.', 'error');
        resetUpload();
    }
}

// SHOW RESULTS
function showResult(data) {
    console.log('📊 Showing result for:', data);
    
    if (window.progressInterval) {
        clearInterval(window.progressInterval);
    }
    
    loadingSection.style.display = 'none';
    resultSection.style.display = 'block';
    
    const isReal = data.prediction === 'REAL';
    
    // Set icon
    if (isReal) {
        resultIcon.innerHTML = '<i class="fas fa-circle-check"></i>';
        resultIcon.className = 'result-icon real';
        resultLabel.textContent = 'AUTHENTIC IMAGE';
        resultLabel.className = 'result-label real';
        resultSubtitle.textContent = 'This image appears to be genuine';
    } else {
        resultIcon.innerHTML = '<i class="fas fa-circle-xmark"></i>';
        resultIcon.className = 'result-icon fake';
        resultLabel.textContent = 'DEEPFAKE DETECTED';
        resultLabel.className = 'result-label fake';
        resultSubtitle.textContent = 'This image may be AI-generated or manipulated';
    }
    
    // Set confidence
    const confidencePercent = data.confidence || 0;
    confidenceValue.textContent = confidencePercent + '%';
    confidenceBar.style.width = confidencePercent + '%';
    confidenceBar.className = 'confidence-bar ' + (isReal ? 'real' : 'fake');
    
    // Set details
    processingTime.textContent = (data.processingTime || 0) + 'ms';
    dataSource.textContent = selectedFile ? selectedFile.name : 'Unknown';
    
    // Show cache info if cached result
    if (data.cached) {
        dataSource.textContent += ' 💾 (Cached)';
        processingTime.textContent += ' ⚡ (From Cache)';
        timestamp.textContent = new Date(data.cacheInfo.firstScan).toLocaleString() + ' (First scan)';
        
        // Show cache hit message
        showToast(`Result retrieved from cache! (${data.cacheInfo.accessCount} total scans)`, 'success');
    } else {
        timestamp.textContent = data.timestamp ? new Date(data.timestamp).toLocaleString() : new Date().toLocaleString();
        showToast(`Analysis complete: ${data.prediction}`, 'success');
    }
    
    resultSection.style.animation = 'fadeInUp 0.5s ease';
    
    console.log('✅ Result displayed successfully');
}

// PROGRESS SIMULATION
function simulateProgress() {
    progressBar.style.width = '0%';
    
    const statuses = [
        'Initializing neural network...',
        'Loading ResNet50 model...',
        'Processing image features...',
        'Analyzing patterns...',
        'Computing confidence scores...',
        'Finalizing results...'
    ];
    
    let progress = 0;
    let statusIndex = 0;
    
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        
        progressBar.style.width = progress + '%';
        
        if (statusIndex < statuses.length) {
            loadingStatus.textContent = statuses[statusIndex];
            statusIndex++;
        }
    }, 500);
    
    // Store interval to clear later if needed
    window.progressInterval = interval;
}

// RESET UPLOAD
function resetUpload() {
    // Clear intervals
    if (window.progressInterval) {
        clearInterval(window.progressInterval);
    }
    
    // Reset file input
    fileInput.value = '';
    selectedFile = null;
    
    // Reset UI
    uploadSection.style.display = 'block';
    previewSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultSection.style.display = 'none';
    analyzeBtn.style.display = 'none';
    
    // Reset progress
    progressBar.style.width = '0%';
    loadingStatus.textContent = 'Initializing neural network...';
}

// STATISTICS
async function showStats() {
    try {
        const response = await fetch(`${API_URL}/predictions`);
        const data = await response.json();
        
        if (data.success && data.predictions) {
            const predictions = data.predictions;
            
            // Calculate stats
            const total = predictions.length;
            const real = predictions.filter(p => p.prediction === 'REAL').length;
            const fake = predictions.filter(p => p.prediction === 'FAKE').length;
            const avgConf = predictions.length > 0
                ? (predictions.reduce((sum, p) => sum + parseFloat(p.confidence), 0) / total).toFixed(1)
                : 0;
            
            // Update display
            totalScans.textContent = total;
            realCount.textContent = real;
            fakeCount.textContent = fake;
            avgConfidence.textContent = avgConf + '%';
            
            // Show panel
            statsPanel.style.display = 'block';
            statsPanel.style.animation = 'fadeIn 0.3s ease';
        } else {
            // Show empty stats
            totalScans.textContent = '0';
            realCount.textContent = '0';
            fakeCount.textContent = '0';
            avgConfidence.textContent = '0%';
            
            statsPanel.style.display = 'block';
        }
    } catch (error) {
        console.error('Error fetching stats:', error);
        showToast('Failed to load statistics', 'error');
    }
}

function hideStats() {
    statsPanel.style.display = 'none';
}

async function updateStats() {
    // Silently update stats in background
    try {
        const response = await fetch(`${API_URL}/predictions`);
        const data = await response.json();
        
        if (data.success && data.predictions) {
            const predictions = data.predictions;
            const total = predictions.length;
            const real = predictions.filter(p => p.prediction === 'REAL').length;
            const fake = predictions.filter(p => p.prediction === 'FAKE').length;
            
            totalScans.textContent = total;
            realCount.textContent = real;
            fakeCount.textContent = fake;
        }
    } catch (error) {
        console.log('Could not update stats');
    }
}


// TOAST NOTIFICATIONS
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');
    const toastIcon = toast.querySelector('.toast-icon');
    
    toastMessage.textContent = message;
    
    // Set icon based on type
    if (type === 'error') {
        toastIcon.className = 'fas fa-circle-xmark toast-icon';
        toast.style.background = 'linear-gradient(135deg, #f44336, #e91e63)';
    } else {
        toastIcon.className = 'fas fa-circle-check toast-icon';
        toast.style.background = 'linear-gradient(135deg, #4caf50, #8bc34a)';
    }
    
    // Show toast
    toast.classList.add('show');
    
    // Hide after 3 seconds
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// VIDEO CONTROLS
function toggleVideo() {
    const video = document.getElementById('bgVideo');
    const icon = document.getElementById('videoIcon');
    
    if (video.paused) {
        video.play();
        icon.className = 'fas fa-pause';
    } else {
        video.pause();
        icon.className = 'fas fa-play';
    }
}

// MOBILE MENU

function toggleMenu() {
    const navRight = document.querySelector('.nav-right');
    navRight.classList.toggle('active');
}
// TYPING ANIMATION


const typingText = document.getElementById('typingText');
const textToType = 'AI-Powered Deepfake Detection';
let charIndex = 0;

function typeText() {
    if (charIndex < textToType.length) {
        typingText.textContent += textToType.charAt(charIndex);
        charIndex++;
        setTimeout(typeText, 100);
    }
}

// Start typing animation on page load
window.addEventListener('load', () => {
    setTimeout(typeText, 500);
});


// INITIALIZE
// Load initial stats
updateStats();

console.log('🚀 Deepfake Detection System Ready');
console.log('📡 API URL:', API_URL);