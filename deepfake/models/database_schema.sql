-- Create database
CREATE DATABASE IF NOT EXISTS deepfake_detection
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE deepfake_db;

-- Drop existing table (for clean reinstall)
DROP TABLE IF EXISTS predictions;

-- ============================================
-- TABLE: predictions
-- ============================================
-- Stores AI prediction results for caching
-- Duplicate images (same hash) return cached results instantly

CREATE TABLE predictions (
    
    -- Primary key
    id INT AUTO_INCREMENT PRIMARY KEY,
    
    -- Image information
    image_name VARCHAR(255) NOT NULL 
        COMMENT 'Original filename of uploaded image',
    
    image_hash VARCHAR(64) NOT NULL 
        COMMENT 'MD5 hash of image content for duplicate detection',
    
    -- Prediction results
    prediction VARCHAR(10) NOT NULL 
        COMMENT 'AI prediction result: FAKE or REAL',
    
    confidence DECIMAL(5,2) NOT NULL 
        COMMENT 'Prediction confidence percentage (0.00-100.00)',
    
    probabilities JSON DEFAULT NULL 
        COMMENT 'Full probability distribution: {"FAKE": 85.5, "REAL": 14.5}',
    
    -- Performance metrics
    processing_time INT DEFAULT NULL 
        COMMENT 'AI processing time in milliseconds',
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
        COMMENT 'Timestamp of first scan',
    
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP 
        COMMENT 'Timestamp of most recent access',
    
    -- Cache statistics
    access_count INT DEFAULT 1 
        COMMENT 'Number of times this image was scanned (cache hits + 1)',
    
    -- Indexes for performance optimization
    UNIQUE KEY unique_hash (image_hash)
        COMMENT 'Ensure each image hash is stored only once',
    
    INDEX idx_prediction (prediction)
        COMMENT 'Fast filtering by prediction type',
    
    INDEX idx_created (created_at)
        COMMENT 'Fast sorting by creation date',
    
    INDEX idx_confidence (confidence)
        COMMENT 'Fast filtering by confidence level'
    
) ENGINE=InnoDB 
  DEFAULT CHARSET=utf8mb4 
  COLLATE=utf8mb4_unicode_ci 
  COMMENT='Deepfake detection results with intelligent caching';

-- ============================================
-- VERIFICATION
-- ============================================

-- Show table structure
DESCRIBE predictions;

-- Show indexes
SHOW INDEX FROM predictions;

-- Verify columns
SELECT 
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE,
    COLUMN_KEY,
    EXTRA,
    COLUMN_COMMENT
FROM information_schema.columns 
WHERE table_schema = 'deepfake_detection' 
AND table_name = 'predictions'
ORDER BY ORDINAL_POSITION;

-- Success message
SELECT '✅ Database schema created successfully!' AS status;
