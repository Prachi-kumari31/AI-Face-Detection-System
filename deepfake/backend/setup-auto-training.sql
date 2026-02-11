USE deepfake_detection;

-- Table to log auto-added training images
CREATE TABLE IF NOT EXISTS training_log (
    id INT AUTO_INCREMENT PRIMARY KEY,
    image_hash VARCHAR(64) UNIQUE,
    image_name VARCHAR(255),
    prediction VARCHAR(10),
    confidence DECIMAL(5,2),
    auto_added BOOLEAN DEFAULT FALSE,
    manually_verified BOOLEAN DEFAULT FALSE,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    verified_at TIMESTAMP NULL,
    INDEX idx_added_at (added_at),
    INDEX idx_auto_added (auto_added),
    INDEX idx_image_hash (image_hash)
);

-- Table to track training history
CREATE TABLE IF NOT EXISTS training_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fake_count INT,
    real_count INT,
    total_count INT,
    balance_ratio DECIMAL(5,2),
    epochs INT DEFAULT 10,
    batch_size INT DEFAULT 32,
    learning_rate DECIMAL(10,8) DEFAULT 0.001,
    accuracy DECIMAL(5,2) NULL,
    loss DECIMAL(10,6) NULL,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    auto_triggered BOOLEAN DEFAULT FALSE,
    duration_seconds INT NULL,
    status VARCHAR(50) DEFAULT 'completed',
    error_message TEXT NULL,
    INDEX idx_trained_at (trained_at),
    INDEX idx_auto_triggered (auto_triggered)
);

-- Update predictions table to track auto-training
SHOW COLUMNS FROM predictions LIKE 'auto_added_to_training';
ALTER TABLE predictions 
ADD COLUMN IF NOT EXISTS auto_added_to_training BOOLEAN DEFAULT FALSE AFTER access_count;

-- View for training statistics
CREATE OR REPLACE VIEW training_stats AS
SELECT 
    COUNT(*) as total_trainings,
    SUM(CASE WHEN auto_triggered = 1 THEN 1 ELSE 0 END) as auto_trainings,
    SUM(CASE WHEN auto_triggered = 0 THEN 1 ELSE 0 END) as manual_trainings,
    AVG(accuracy) as avg_accuracy,
    MAX(trained_at) as last_training_date,
    DATEDIFF(NOW(), MAX(trained_at)) as days_since_last_training
FROM training_history;

-- View for training data summary
CREATE OR REPLACE VIEW training_data_summary AS
SELECT 
    COUNT(*) as total_images,
    SUM(CASE WHEN auto_added = 1 THEN 1 ELSE 0 END) as auto_added_count,
    SUM(CASE WHEN manually_verified = 1 THEN 1 ELSE 0 END) as verified_count,
    SUM(CASE WHEN prediction = 'FAKE' THEN 1 ELSE 0 END) as fake_count,
    SUM(CASE WHEN prediction = 'REAL' THEN 1 ELSE 0 END) as real_count,
    AVG(confidence) as avg_confidence,
    MIN(added_at) as first_added,
    MAX(added_at) as last_added
FROM training_log;

SELECT 'Auto-training tables created successfully!' as status;

-- Show current state
SELECT * FROM training_stats;
SELECT * FROM training_data_summary;