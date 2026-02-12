const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs').promises;
const { PythonShell } = require('python-shell');
const pool = require('../database');
const ImageProcessor = require('./utils/imageProcessor');

// Configure multer for file upload
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, 'uploads/');
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname);
  }
});

const upload = multer({ storage: storage });

// Prediction endpoint
router.post('/predict', upload.single('image'), async (req, res) => {
  const startTime = Date.now();
  let imagePath = null;

  try {
    // Validate image
    if (!req.file) {
      return res.status(400).json({ error: 'No image provided' });
    }

    const validation = ImageProcessor.validateImage(req.file);
    if (!validation.valid) {
      await fs.unlink(req.file.path);
      return res.status(400).json({ error: validation.error });
    }

    imagePath = req.file.path;
    const imageHash = await ImageProcessor.generateHash(imagePath);

    // Check if image already processed
    const [existing] = await pool.query(
      'SELECT * FROM predictions WHERE image_hash = ?',
      [imageHash]
    );

    if (existing.length > 0) {
      const result = existing[0];
      const processingTime = Date.now() - startTime;

      // Log the query
      await pool.query(
        'INSERT INTO detection_logs (prediction_id, processing_time_ms, from_cache) VALUES (?, ?, ?)',
        [result.id, processingTime, true]
      );

      // Clean up uploaded file
      await fs.unlink(imagePath);

      return res.json({
        prediction: result.prediction,
        confidence: result.confidence,
        fromCache: true,
        processingTime: processingTime,
        timestamp: result.created_at
      });
    }

    // Preprocess image
    const preprocessedPath = `uploads/processed-${Date.now()}.jpg`;
    await ImageProcessor.preprocessImage(imagePath, preprocessedPath);

    // Run Python prediction script
    const options = {
      mode: 'json',
      pythonPath: 'python3',
      pythonOptions: ['-u'],
      scriptPath: path.join(__dirname, '../python'),
      args: [preprocessedPath]
    };

    const results = await PythonShell.run('predict.py', options);
    const prediction = results[0];

    // Save to database
    const [insertResult] = await pool.query(
      'INSERT INTO predictions (image_hash, image_name, prediction, confidence) VALUES (?, ?, ?, ?)',
      [imageHash, req.file.originalname, prediction.label, prediction.confidence]
    );

    const processingTime = Date.now() - startTime;

    // Log the prediction
    await pool.query(
      'INSERT INTO detection_logs (prediction_id, processing_time_ms, from_cache) VALUES (?, ?, ?)',
      [insertResult.insertId, processingTime, false]
    );

    // Clean up files
    await fs.unlink(imagePath);
    await fs.unlink(preprocessedPath);

    res.json({
      prediction: prediction.label,
      confidence: prediction.confidence,
      fromCache: false,
      processingTime: processingTime,
      timestamp: new Date()
    });

  } catch (error) {
    console.error('Prediction error:', error);
    
    // Clean up files
    if (imagePath) {
      try {
        await fs.unlink(imagePath);
      } catch (e) {}
    }

    res.status(500).json({
      error: 'Prediction failed',
      message: error.message
    });
  }
});

// Get prediction history
router.get('/history', async (req, res) => {
  try {
    const [results] = await pool.query(
      'SELECT * FROM predictions ORDER BY created_at DESC LIMIT 50'
    );
    res.json(results);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Get statistics
router.get('/stats', async (req, res) => {
  try {
    const [stats] = await pool.query(`
      SELECT 
        COUNT(*) as total_predictions,
        SUM(CASE WHEN prediction = 'REAL' THEN 1 ELSE 0 END) as real_count,
        SUM(CASE WHEN prediction = 'FAKE' THEN 1 ELSE 0 END) as fake_count,
        AVG(confidence) as avg_confidence
      FROM predictions
    `);

    const [cacheStats] = await pool.query(`
      SELECT 
        SUM(CASE WHEN from_cache = 1 THEN 1 ELSE 0 END) as cache_hits,
        SUM(CASE WHEN from_cache = 0 THEN 1 ELSE 0 END) as cache_misses,
        AVG(processing_time_ms) as avg_processing_time
      FROM detection_logs
    `);

    res.json({
      ...stats[0],
      ...cacheStats[0]
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;
