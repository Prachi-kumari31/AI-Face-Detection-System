// require('dotenv').config();
// const express = require('express');
// const multer = require('multer');
// const { spawn } = require('child_process');
// const path = require('path');
// const fs = require('fs');
// const cors = require('cors');
// const db = require('./database');
// const { generateImageHash } = require('./utils/imageHash');

// const app = express();

// // Middleware
// app.use(cors());
// app.use(express.json());
// app.use(express.urlencoded({ extended: true }));
// app.use(express.static(path.join(__dirname, '..', 'frontend')));

// // Multer configuration
// const storage = multer.diskStorage({
//     destination: function (req, file, cb) {
//         const uploadDir = path.join(__dirname, 'uploads');
//         if (!fs.existsSync(uploadDir)) {
//             fs.mkdirSync(uploadDir, { recursive: true });
//         }
//         cb(null, uploadDir);
//     },
//     filename: function (req, file, cb) {
//         const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
//         cb(null, 'upload-' + uniqueSuffix + path.extname(file.originalname));
//     }
// });

// const upload = multer({ 
//     storage: storage,
//     limits: { fileSize: 10 * 1024 * 1024 },
//     fileFilter: function (req, file, cb) {
//         const allowedTypes = /jpeg|jpg|png/;
//         const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
//         const mimetype = allowedTypes.test(file.mimetype);
        
//         if (mimetype && extname) {
//             return cb(null, true);
//         } else {
//             cb(new Error('Only JPEG, JPG, and PNG images are allowed'));
//         }
//     }
// });

// // ============================================================================
// // CLEANUP FUNCTIONS - GUARANTEED WORKING VERSION
// // ============================================================================

// async function cleanupDatabaseCache(minutesOld = 60) {
//     try {
//         console.log(`\n🧹 Database Cleanup (${new Date().toLocaleTimeString()})`);
//         console.log(`   Deleting entries older than ${minutesOld} minutes (${Math.floor(minutesOld/60)}h)`);
        
//         // Show entries before deletion
//         const [before] = await db.query(`
//             SELECT 
//                 COUNT(*) as total,
//                 SUM(CASE WHEN TIMESTAMPDIFF(MINUTE, created_at, NOW()) >= ? THEN 1 ELSE 0 END) as old_count
//             FROM predictions
//         `, [minutesOld]);
        
//         console.log(`   Before: ${before[0].total} total, ${before[0].old_count} old entries`);
        
//         if (before[0].old_count > 0) {
//             // Show details of entries to delete
//             const [toDelete] = await db.query(`
//                 SELECT 
//                     id,
//                     image_name,
//                     TIMESTAMPDIFF(MINUTE, created_at, NOW()) as age_min
//                 FROM predictions 
//                 WHERE TIMESTAMPDIFF(MINUTE, created_at, NOW()) >= ?
//                 ORDER BY created_at ASC
//                 LIMIT 5
//             `, [minutesOld]);
            
//             console.log(`   Deleting entries:`);
//             toDelete.forEach(e => {
//                 console.log(`      - ID ${e.id}: ${e.image_name} (${Math.floor(e.age_min/60)}h ${e.age_min%60}m)`);
//             });
//         }
        
//         // DELETE using TIMESTAMPDIFF with MINUTES
//         const [result] = await db.query(`
//             DELETE FROM predictions 
//             WHERE TIMESTAMPDIFF(MINUTE, created_at, NOW()) >= ?
//         `, [minutesOld]);
        
//         if (result.affectedRows > 0) {
//             console.log(`   ✅ Deleted ${result.affectedRows} entries`);
//         } else {
//             console.log(`   ℹ️  No old entries found`);
//         }
        
//         // Show remaining count
//         const [after] = await db.query('SELECT COUNT(*) as total FROM predictions');
//         console.log(`   After: ${after[0].total} entries remaining\n`);
        
//         return result.affectedRows;
        
//     } catch (error) {
//         console.error('❌ Database cleanup error:', error.message);
//         return 0;
//     }
// }
// function cleanupUploadedFiles(hoursOld = 0.05) {  // ✅ Changed default to 3 minutes
//     const uploadDir = path.join(__dirname, 'uploads');
//     const now = Date.now();
//     const maxAge = hoursOld * 60 * 60 * 1000;
    
//     console.log(`\n🧹 File Cleanup (${new Date().toLocaleTimeString()})`);
//     console.log(`   Target: Delete files older than ${hoursOld} hours (${(hoursOld * 60).toFixed(1)} minutes)`);
//     console.log(`   Folder: ${uploadDir}`);
    
//     try {
//         if (!fs.existsSync(uploadDir)) {
//             console.log('   ❌ Upload directory does not exist\n');
//             return 0;
//         }
        
//         const files = fs.readdirSync(uploadDir);
//         console.log(`   📂 Found: ${files.length} file(s)`);
        
//         if (files.length === 0) {
//             console.log('   ℹ️  Folder is empty\n');
//             return 0;
//         }
        
//         let deletedCount = 0;
//         let totalSize = 0;
//         let keptCount = 0;
        
//         files.forEach(file => {
//             try {
//                 const filePath = path.join(uploadDir, file);
//                 const stats = fs.statSync(filePath);
//                 const fileAge = now - stats.mtimeMs;
//                 const ageMinutes = (fileAge / (1000 * 60)).toFixed(1);
//                 const ageHours = (fileAge / (1000 * 60 * 60)).toFixed(2);
                
//                 console.log(`   Checking: ${file} (${ageMinutes} min old)`);
                
//                 if (fileAge >= maxAge) {
//                     // Delete old file
//                     totalSize += stats.size;
//                     fs.unlinkSync(filePath);
//                     deletedCount++;
//                     console.log(`      🗑️  DELETED (age: ${ageMinutes} min)`);
//                 } else {
//                     // Keep file
//                     keptCount++;
//                     console.log(`      ✅ KEPT (too new)`);
//                 }
//             } catch (err) {
//                 console.warn(`   ⚠️  Error with ${file}: ${err.message}`);
//             }
//         });
        
//         if (deletedCount > 0) {
//             console.log(`\n   ✅ Deleted: ${deletedCount} file(s)`);
//             console.log(`   💾 Freed: ${(totalSize / 1024).toFixed(2)} KB`);
//         } else {
//             console.log(`\n   ℹ️  No files old enough to delete`);
//         }
        
//         if (keptCount > 0) {
//             console.log(`   📦 Kept: ${keptCount} recent file(s)`);
//         }
        
//         console.log('');
//         return deletedCount;
        
//     } catch (err) {
//         console.error('❌ File cleanup error:', err.message);
//         console.error('Stack:', err.stack);
//         return 0;
//     }
// }
// async function performCompleteCleanup() {
//     console.log('\n' + '='.repeat(60));
//     console.log('🧹 AUTOMATIC CLEANUP');
//     console.log('='.repeat(60));
//     console.log(`⏰ ${new Date().toLocaleString()}`);
    
//     // Get retention settings from environment
//     const fileRetentionHours = parseInt(process.env.FILE_RETENTION_HOURS) || 24;
//     const cacheRetentionMinutes = (parseInt(process.env.CACHE_RETENTION_HOURS) || 48) * 60;
    
//     console.log(`\n📋 Retention Policy:`);
//     console.log(`   Files: ${fileRetentionHours} hours`);
//     console.log(`   Cache: ${cacheRetentionMinutes / 60} hours\n`);
    
//     // Cleanup old files (using configured retention)
//     const filesDeleted = cleanupUploadedFiles(fileRetentionHours);
    
//     // Cleanup old cache (using configured retention)
//     const cacheDeleted = await cleanupDatabaseCache(cacheRetentionMinutes);
    
//     console.log('='.repeat(60));
//     console.log(`📊 Summary: ${filesDeleted} files, ${cacheDeleted} cache entries deleted`);
//     console.log('='.repeat(60) + '\n');
// }

// function startCleanupScheduler() {
//     // Use parseFloat for fractional hours
//     const intervalMinutes = parseInt(process.env.CLEANUP_INTERVAL_MINUTES) || 1;
//     const fileRetentionHours = parseFloat(process.env.FILE_RETENTION_HOURS) || 0.05;
//     const cacheRetentionHours = parseFloat(process.env.CACHE_RETENTION_HOURS) || 0.1;
    
//     console.log('\n' + '='.repeat(60));
//     console.log('🤖 CLEANUP SCHEDULER STARTING');
//     console.log('='.repeat(60));
//     console.log(`⏰ Interval: Every ${intervalMinutes} minute(s)`);
//     console.log(`📁 File retention: ${fileRetentionHours} hours (${(fileRetentionHours * 60).toFixed(1)} minutes)`);
//     console.log(`🗄️  Cache retention: ${cacheRetentionHours} hours (${(cacheRetentionHours * 60).toFixed(1)} minutes)`);
//     console.log(`🚀 First cleanup: In 10 seconds`);
//     console.log('='.repeat(60) + '\n');
    
//     // Run first cleanup after 10 seconds
//     setTimeout(async () => {
//         console.log('🚀 Running initial cleanup...');
//         await performCompleteCleanup();
//     }, 10000);
    
//     // Then run every X minutes
//     const intervalMs = intervalMinutes * 60 * 1000;
//     setInterval(async () => {
//         await performCompleteCleanup();
//     }, intervalMs);
    
//     console.log('✅ Scheduler is ACTIVE and RUNNING!\n');
// }

// // ============================================================================
// // API ENDPOINTS
// // ============================================================================

// app.get('/api/health', (req, res) => {
//     res.json({ 
//         status: 'ok', 
//         timestamp: new Date().toISOString(),
//         uptime: Math.round(process.uptime())
//     });
// });

// app.post('/api/predict', upload.single('image'), async (req, res) => {
//     console.log('\n' + '='.repeat(60));
//     console.log('📸 NEW PREDICTION REQUEST');
//     console.log('='.repeat(60));
    
//     if (!req.file) {
//         console.log('❌ No file');
//         console.log('='.repeat(60) + '\n');
//         return res.status(400).json({ success: false, error: 'No file' });
//     }

//     const imagePath = req.file.path;
//     const imageName = req.file.originalname;
//     const startTime = Date.now();

//     console.log(`📁 File: ${imageName}`);
//     console.log(`📏 Size: ${(req.file.size / 1024).toFixed(2)} KB`);

//     try {
//         console.log('🔐 Generating hash...');
//         const imageHash = await generateImageHash(imagePath);
//         console.log(`   Hash: ${imageHash}`);
        
//         console.log('🔍 Checking cache...');
//         const [cached] = await db.query('SELECT * FROM predictions WHERE image_hash = ?', [imageHash]);
        
//       if (cached.length > 0) {
//     console.log('✅ CACHE HIT');
//     console.log(`   Result: ${cached[0].prediction} (${cached[0].confidence}%)`);
//     console.log(`   Age: ${Math.round((Date.now() - new Date(cached[0].created_at).getTime()) / 60000)} min`);
    
//     // Update access info
//     await db.query(
//         'UPDATE predictions SET last_accessed = NOW(), access_count = access_count + 1 WHERE image_hash = ?', 
//         [imageHash]
//     );
    
//     // ✅ IMPORTANT: Keep file, don't delete (scheduler will handle it)
//     console.log('💾 File kept in uploads/ - will auto-delete after retention period');
    
//     const processingTime = Date.now() - startTime;
//     console.log(`⏱️  ${processingTime}ms (from cache)`);
//     console.log('='.repeat(60) + '\n');
    
//     // Handle probabilities
//     let probabilities;
//     try {
//         if (typeof cached[0].probabilities === 'string') {
//             probabilities = JSON.parse(cached[0].probabilities);
//         } else if (typeof cached[0].probabilities === 'object') {
//             probabilities = cached[0].probabilities;
//         } else {
//             const conf = parseFloat(cached[0].confidence);
//             probabilities = cached[0].prediction === 'FAKE' 
//                 ? { FAKE: conf, REAL: 100 - conf }
//                 : { REAL: conf, FAKE: 100 - conf };
//         }
//     } catch (err) {
//         const conf = parseFloat(cached[0].confidence);
//         probabilities = cached[0].prediction === 'FAKE' 
//             ? { FAKE: conf, REAL: 100 - conf }
//             : { REAL: conf, FAKE: 100 - conf };
//     }
    
//     return res.json({
//         success: true,
//         prediction: cached[0].prediction,
//         confidence: parseFloat(cached[0].confidence),
//         probabilities: probabilities,
//         processingTime: processingTime,
//         timestamp: cached[0].created_at,
//         cached: true
//     });
// }

//         console.log('❌ CACHE MISS - Running AI...');
        
//         const pythonScript = path.join(__dirname, 'python', 'prediction.py');
        
//         if (!fs.existsSync(pythonScript)) {
//             console.error(`❌ Python script not found: ${pythonScript}`);
//             throw new Error('AI script not found');
//         }
        
//         const python = spawn('python', [pythonScript, imagePath]);
//         let dataString = '';
//         let errorString = '';

//         python.stdout.on('data', (data) => { dataString += data.toString(); });
//         python.stderr.on('data', (data) => { 
//             errorString += data.toString();
//             const line = data.toString().trim();
//             if (line) console.log('[Python]', line);
//         });

//        python.on('close', async (code) => {
//     // ✅ NEW CODE - Keep file, don't delete
//     console.log('💾 File kept in uploads/ folder (will auto-delete after retention period)');
    
//     // File will be deleted by cleanup scheduler after configured hours

//             const processingTime = Date.now() - startTime;

//             if (code !== 0) {
//                 console.error('❌ Python failed');
//                 console.log('='.repeat(60) + '\n');
//                 return res.status(500).json({ success: false, error: 'Prediction failed' });
//             }

//             try {
//                 const jsonStart = dataString.indexOf('{');
//                 if (jsonStart === -1) throw new Error('No JSON');
//                 const result = JSON.parse(dataString.substring(jsonStart));

//                 if (!result.success) throw new Error(result.error || 'Failed');

//                 console.log('✅ Prediction successful');
//                 console.log(`   Result: ${result.prediction} (${result.confidence}%)`);

//                 try {
//                     await db.query(
//                         `INSERT INTO predictions 
//                         (image_name, image_hash, prediction, confidence, probabilities, processing_time, created_at, last_accessed, access_count) 
//                         VALUES (?, ?, ?, ?, ?, ?, NOW(), NOW(), 1)`,
//                         [imageName, imageHash, result.prediction, result.confidence, JSON.stringify(result.probabilities), processingTime]
//                     );
//                     console.log('💾 Cached');
//                 } catch (dbError) {
//                     console.warn('⚠️  Not cached');
//                 }

//                 console.log(`⏱️  ${processingTime}ms`);
//                 console.log('='.repeat(60) + '\n');
                
//                 res.json({
//                     success: true,
//                     prediction: result.prediction,
//                     confidence: result.confidence,
//                     probabilities: result.probabilities,
//                     processingTime: processingTime,
//                     timestamp: new Date().toISOString(),
//                     cached: false
//                 });
                
//             } catch (err) {
//                 console.error('❌ Parse error');
//                 console.log('='.repeat(60) + '\n');
//                 res.status(500).json({ success: false, error: 'Parse error' });
//             }
//         });

//     } catch (error) {
//         console.error('❌ Error:', error.message);
//         console.log('='.repeat(60) + '\n');
        
//         try { 
//             if (fs.existsSync(imagePath)) fs.unlinkSync(imagePath); 
//         } catch (e) {}
        
//         res.status(500).json({ success: false, error: error.message });
//     }
// });

// app.get('/api/predictions', async (req, res) => {
//     try {
//         const [rows] = await db.query(`
//             SELECT 
//                 *,
//                 TIMESTAMPDIFF(MINUTE, created_at, NOW()) as age_minutes
//             FROM predictions 
//             ORDER BY last_accessed DESC 
//             LIMIT 100
//         `);
        
//         res.json({ success: true, count: rows.length, predictions: rows });
//     } catch (error) {
//         res.status(500).json({ success: false, error: error.message });
//     }
// });

// app.get('/api/admin/cleanup-now', async (req, res) => {
//     try {
//         console.log('\n🧹 Manual cleanup via API\n');
        
//         const filesDeleted = cleanupUploadedFiles(1);
//         const cacheDeleted = await cleanupDatabaseCache(60);
        
//         res.json({
//             success: true,
//             filesDeleted: filesDeleted,
//             cacheDeleted: cacheDeleted,
//             timestamp: new Date().toISOString()
//         });
//     } catch (error) {
//         res.status(500).json({ success: false, error: error.message });
//     }
// });

// app.get('/api/admin/cache-stats', async (req, res) => {
//     try {
//         const [stats] = await db.query(`
//             SELECT 
//                 COUNT(*) as total,
//                 COUNT(CASE WHEN prediction = 'FAKE' THEN 1 END) as fake,
//                 COUNT(CASE WHEN prediction = 'REAL' THEN 1 END) as real,
//                 AVG(confidence) as avg_confidence
//             FROM predictions
//         `);
        
//         res.json({ success: true, stats: stats[0] });
//     } catch (error) {
//         res.status(500).json({ success: false, error: error.message });
//     }
// });

// app.get('*', (req, res) => {
//     res.sendFile(path.join(__dirname, '..', 'frontend', 'index.html'));
// });

// // ============================================================================
// // START SERVER
// // ============================================================================

// const PORT = parseInt(process.env.PORT) || 3000;

// app.listen(PORT, () => {
//     console.log('\n' + '='.repeat(60));
//     console.log('🚀 DEEPFAKE DETECTION SERVER');
//     console.log('='.repeat(60));
//     console.log(`📡 Server:  http://localhost:${PORT}`);
//     console.log(`🔍 Predict: http://localhost:${PORT}/api/predict`);
//     console.log(`🧹 Cleanup: http://localhost:${PORT}/api/admin/cleanup-now`);
//     console.log('='.repeat(60) + '\n');
    
//     setTimeout(() => {
//         startCleanupScheduler();
//     }, 2000);
// });








require('dotenv').config();
const express = require('express');
const multer = require('multer');
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const cors = require('cors');
const db = require('./database');
const { generateImageHash } = require('./utils/imageHash');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, '..', 'frontend')));

// Multer configuration
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        const uploadDir = path.join(__dirname, 'uploads');
        if (!fs.existsSync(uploadDir)) {
            fs.mkdirSync(uploadDir, { recursive: true });
        }
        cb(null, uploadDir);
    },
    filename: function (req, file, cb) {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, 'upload-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({ 
    storage: storage,
    limits: { fileSize: 10 * 1024 * 1024 },
    fileFilter: function (req, file, cb) {
        const allowedTypes = /jpeg|jpg|png/;
        const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
        const mimetype = allowedTypes.test(file.mimetype);
        
        if (mimetype && extname) {
            return cb(null, true);
        } else {
            cb(new Error('Only JPEG, JPG, and PNG images are allowed'));
        }
    }
});

// ============================================================================
// AUTOMATIC TRAINING PIPELINE FUNCTIONS
// ============================================================================

/**
 * Automatically add high-confidence predictions to training data
 */
async function autoAddToTrainingData(imagePath, imageName, prediction, confidence, imageHash) {
    try {
        const autoTrainingEnabled = process.env.AUTO_TRAINING_ENABLED === 'true';
        const highConfThreshold = parseFloat(process.env.HIGH_CONFIDENCE_THRESHOLD) || 95;
        
        if (!autoTrainingEnabled) {
            console.log('ℹ️  Auto-training disabled');
            return false;
        }
        
        if (confidence >= highConfThreshold) {
            const trainingBaseDir = path.join(__dirname, 'training_data');
            const targetDir = path.join(trainingBaseDir, prediction);
            
            // Create folder if doesn't exist
            if (!fs.existsSync(trainingBaseDir)) {
                fs.mkdirSync(trainingBaseDir, { recursive: true });
            }
            if (!fs.existsSync(targetDir)) {
                fs.mkdirSync(targetDir, { recursive: true });
            }
            
            // Generate new filename
            const timestamp = Date.now();
            const ext = path.extname(imagePath);
            const hashShort = imageHash.substring(0, 8);
            const newFilename = `auto_${prediction.toLowerCase()}_${timestamp}_${hashShort}${ext}`;
            const targetPath = path.join(targetDir, newFilename);
            
            // Copy to training data (not move, original stays in uploads for retention period)
            fs.copyFileSync(imagePath, targetPath);
            
            console.log(`✅ Auto-added to training: ${prediction} (${confidence}%) → ${newFilename}`);
            
            // Log to database
            try {
                await db.query(
                    `INSERT INTO training_log 
                    (image_hash, image_name, prediction, confidence, auto_added, added_at) 
                    VALUES (?, ?, ?, ?, 1, NOW())
                    ON DUPLICATE KEY UPDATE 
                    prediction = VALUES(prediction),
                    confidence = VALUES(confidence),
                    auto_added = 1,
                    added_at = NOW()`,
                    [imageHash, imageName, prediction, confidence]
                );
            } catch (dbErr) {
                console.warn('⚠️  Could not log to training_log:', dbErr.message);
            }
            
            return true;
        } else {
            console.log(`ℹ️  Not auto-added: ${prediction} (${confidence}% < ${highConfThreshold}%)`);
            return false;
        }
        
    } catch (error) {
        console.error('❌ Auto-add error:', error.message);
        return false;
    }
}

/**
 * Check if automatic training should trigger
 */
async function checkAndTriggerAutoTraining() {
    try {
        const autoTrainingEnabled = process.env.AUTO_TRAINING_ENABLED === 'true';
        
        if (!autoTrainingEnabled) {
            console.log('ℹ️  Auto-training disabled in config');
            return false;
        }
        
        const MIN_NEW_IMAGES = parseInt(process.env.MIN_TRAINING_IMAGES) || 50;
        const MIN_BALANCE_RATIO = parseFloat(process.env.MIN_BALANCE_RATIO) || 0.3;
        const MIN_DAYS_BETWEEN = parseFloat(process.env.MIN_DAYS_BETWEEN_TRAINING) || 1;
        
        const trainingDir = path.join(__dirname, 'training_data');
        const fakeDir = path.join(trainingDir, 'FAKE');
        const realDir = path.join(trainingDir, 'REAL');
        
        // Count images
        let fakeCount = 0;
        let realCount = 0;
        
        if (fs.existsSync(fakeDir)) {
            fakeCount = fs.readdirSync(fakeDir).filter(f => /\.(jpg|jpeg|png)$/i.test(f)).length;
        }
        if (fs.existsSync(realDir)) {
            realCount = fs.readdirSync(realDir).filter(f => /\.(jpg|jpeg|png)$/i.test(f)).length;
        }
        
        const totalCount = fakeCount + realCount;
        
        // Check last training time
        const [lastTraining] = await db.query(
            'SELECT MAX(trained_at) as last_trained FROM training_history'
        );
        
        const lastTrainedDate = lastTraining[0]?.last_trained;
        const daysSinceTraining = lastTrainedDate 
            ? (Date.now() - new Date(lastTrainedDate).getTime()) / (1000 * 60 * 60 * 24)
            : 999;
        
        // Calculate balance
        const balance = totalCount > 0 ? Math.min(fakeCount, realCount) / totalCount : 0;
        
        console.log(`\n🤖 Auto-Training Check (${new Date().toLocaleTimeString()}):`);
        console.log(`   FAKE: ${fakeCount}, REAL: ${realCount}, Total: ${totalCount}`);
        console.log(`   Balance ratio: ${(balance * 100).toFixed(1)}% (min: ${(MIN_BALANCE_RATIO * 100).toFixed(1)}%)`);
        console.log(`   Days since last training: ${daysSinceTraining.toFixed(1)} (min: ${MIN_DAYS_BETWEEN})`);
        console.log(`   Min images required: ${MIN_NEW_IMAGES}`);
        
        // Decide if should train
        const hasEnoughImages = totalCount >= MIN_NEW_IMAGES;
        const isBalanced = balance >= MIN_BALANCE_RATIO;
        const hasWaitedLongEnough = daysSinceTraining >= MIN_DAYS_BETWEEN;
        const hasMinimumOfEach = fakeCount >= 10 && realCount >= 10;
        
        console.log(`   Criteria check:`);
        console.log(`      Enough images (${totalCount} >= ${MIN_NEW_IMAGES}): ${hasEnoughImages ? '✅' : '❌'}`);
        console.log(`      Balanced (${(balance * 100).toFixed(1)}% >= ${(MIN_BALANCE_RATIO * 100).toFixed(1)}%): ${isBalanced ? '✅' : '❌'}`);
        console.log(`      Waited enough (${daysSinceTraining.toFixed(1)}d >= ${MIN_DAYS_BETWEEN}d): ${hasWaitedLongEnough ? '✅' : '❌'}`);
        console.log(`      Min 10 of each: ${hasMinimumOfEach ? '✅' : '❌'}`);
        
        const shouldTrain = hasEnoughImages && isBalanced && hasWaitedLongEnough && hasMinimumOfEach;
        
        if (shouldTrain) {
            console.log(`   ✅ All criteria met - Triggering automatic training!\n`);
            await triggerAutomaticTraining(fakeCount, realCount, totalCount, balance);
            return true;
        } else {
            console.log(`   ℹ️  Not ready for training yet\n`);
            return false;
        }
        
    } catch (error) {
        console.error('❌ Auto-training check error:', error.message);
        return false;
    }
}

/**
 * Trigger automatic model training
 */
async function triggerAutomaticTraining(fakeCount, realCount, totalCount, balance) {
    const startTime = Date.now();
    
    try {
        console.log('\n' + '='.repeat(60));
        console.log('🎓 AUTOMATIC MODEL TRAINING STARTED');
        console.log('='.repeat(60));
        console.log(`Started at: ${new Date().toLocaleString()}`);
        
        const trainingParams = {
            epochs: 10,
            batchSize: 32,
            learningRate: 0.001,
            fakeCount: fakeCount,
            realCount: realCount,
            totalCount: totalCount,
            balanceRatio: balance
        };
        
        console.log('\nTraining Parameters:');
        console.log(JSON.stringify(trainingParams, null, 2));
        console.log('');
        
        // ✅ REAL TRAINING: Call Python script
        const pythonScript = path.join(__dirname, 'python', 'train_model.py');
        const trainingDataDir = path.join(__dirname, 'training_data');
        const modelPath = path.join(__dirname, '..', 'models', 'deepfake_detector_cnn.pth');
        
        console.log('🐍 Calling Python training script...');
        console.log(`   Script: ${pythonScript}`);
        console.log(`   Data: ${trainingDataDir}`);
        console.log(`   Model: ${modelPath}\n`);
        
        // Spawn Python process
        const python = spawn('python', [
            pythonScript,
            '--data_dir', trainingDataDir,
            '--model_path', modelPath,
            '--epochs', trainingParams.epochs.toString(),
            '--batch_size', trainingParams.batchSize.toString(),
            '--learning_rate', trainingParams.learningRate.toString()
        ]);
        
        let pythonOutput = '';
        let trainingLogs = [];
        
        // Capture stdout (JSON result)
        python.stdout.on('data', (data) => {
            pythonOutput += data.toString();
        });
        
        // Capture stderr (training logs)
        python.stderr.on('data', (data) => {
            const log = data.toString().trim();
            if (log) {
                console.log('[Python]', log);
                trainingLogs.push(log);
            }
        });
        
        // Wait for Python to complete
        const pythonResult = await new Promise((resolve, reject) => {
            python.on('close', (code) => {
                if (code === 0) {
                    try {
                        // Parse JSON result
                        const jsonStart = pythonOutput.indexOf('{');
                        if (jsonStart !== -1) {
                            const result = JSON.parse(pythonOutput.substring(jsonStart));
                            resolve(result);
                        } else {
                            reject(new Error('No JSON output from Python'));
                        }
                    } catch (e) {
                        reject(new Error(`Failed to parse Python output: ${e.message}`));
                    }
                } else {
                    reject(new Error(`Python script failed with code ${code}`));
                }
            });
            
            python.on('error', (error) => {
                reject(new Error(`Failed to start Python: ${error.message}`));
            });
        });
        
        const duration = Math.round((Date.now() - startTime) / 1000);
        
        console.log('\n✅ Training completed successfully!');
        console.log(`   Accuracy: ${pythonResult.accuracy}%`);
        console.log(`   Duration: ${duration} seconds`);
        console.log(`   Samples: ${pythonResult.samples_trained}`);
        
        // ✅ Cleanup training files
        console.log('\n🗑️  Cleaning up training data...');
        const deletedFiles = await deleteTrainingFilesAfterTraining();
        console.log(`   ✅ Deleted ${deletedFiles} training files\n`);
        
        // Log to database
        await db.query(
            `INSERT INTO training_history 
            (fake_count, real_count, total_count, balance_ratio, epochs, batch_size, 
             learning_rate, accuracy, trained_at, auto_triggered, duration_seconds, 
             status, files_deleted) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NOW(), 1, ?, 'completed', ?)`,
            [
                fakeCount, 
                realCount, 
                totalCount,
                balance,
                trainingParams.epochs, 
                trainingParams.batchSize,
                trainingParams.learningRate,
                pythonResult.accuracy,
                duration,
                deletedFiles
            ]
        );
        
        console.log('💾 Training logged to database');
        console.log('='.repeat(60) + '\n');
        
        return true;
        
    } catch (error) {
        const duration = Math.round((Date.now() - startTime) / 1000);
        
        console.error('❌ Training error:', error.message);
        
        // Log error to database
        try {
            await db.query(
                `INSERT INTO training_history 
                (fake_count, real_count, total_count, balance_ratio, trained_at, 
                 auto_triggered, duration_seconds, status, error_message) 
                VALUES (?, ?, ?, ?, NOW(), 1, ?, 'failed', ?)`,
                [fakeCount, realCount, totalCount, balance, duration, error.message]
            );
        } catch (dbErr) {}
        
        console.log('='.repeat(60) + '\n');
        return false;
    }
}
/**
 * Delete training files after successful training
 */
async function deleteTrainingFilesAfterTraining() {
    try {
        const trainingDir = path.join(__dirname, 'training_data');
        const fakeDir = path.join(trainingDir, 'FAKE');
        const realDir = path.join(trainingDir, 'REAL');
        
        let deletedCount = 0;
        
        // Delete FAKE images
        if (fs.existsSync(fakeDir)) {
            const fakeFiles = fs.readdirSync(fakeDir);
            fakeFiles.forEach(file => {
                try {
                    const filePath = path.join(fakeDir, file);
                    fs.unlinkSync(filePath);
                    deletedCount++;
                } catch (e) {
                    console.warn(`   ⚠️  Could not delete ${file}: ${e.message}`);
                }
            });
            console.log(`   Deleted ${fakeFiles.length} FAKE images`);
        }
        
        // Delete REAL images
        if (fs.existsSync(realDir)) {
            const realFiles = fs.readdirSync(realDir);
            realFiles.forEach(file => {
                try {
                    const filePath = path.join(realDir, file);
                    fs.unlinkSync(filePath);
                    deletedCount++;
                } catch (e) {
                    console.warn(`   ⚠️  Could not delete ${file}: ${e.message}`);
                }
            });
            console.log(`   Deleted ${realFiles.length} REAL images`);
        }
        
        // Clear training_log table
        try {
            await db.query('DELETE FROM training_log');
            console.log(`   Cleared training_log database`);
        } catch (dbErr) {
            console.warn(`   ⚠️  Could not clear training_log: ${dbErr.message}`);
        }
        
        return deletedCount;
        
    } catch (error) {
        console.error('   ❌ Error deleting training files:', error.message);
        return 0;
    }
}

/**
 * Schedule automatic training checks
 */
function startAutoTrainingScheduler() {
    const autoTrainingEnabled = process.env.AUTO_TRAINING_ENABLED === 'true';
    
    if (!autoTrainingEnabled) {
        console.log('\n⚠️  Auto-training is DISABLED in .env file\n');
        return;
    }
    
    const CHECK_INTERVAL_HOURS = parseInt(process.env.TRAINING_CHECK_INTERVAL_HOURS) || 6;
    
    console.log('\n' + '='.repeat(60));
    console.log('🤖 AUTO-TRAINING SCHEDULER');
    console.log('='.repeat(60));
    console.log(`⏰ Check interval: Every ${CHECK_INTERVAL_HOURS} hour(s)`);
    console.log(`📊 Min images: ${process.env.MIN_TRAINING_IMAGES || 50}`);
    console.log(`⚖️  Min balance: ${(parseFloat(process.env.MIN_BALANCE_RATIO || 0.3) * 100).toFixed(0)}%`);
    console.log(`⏳ Min days between training: ${process.env.MIN_DAYS_BETWEEN_TRAINING || 1}`);
    console.log(`🎯 Confidence threshold: ${process.env.HIGH_CONFIDENCE_THRESHOLD || 95}%`);
    console.log(`🚀 First check: In 1 minute`);
    console.log('='.repeat(60) + '\n');
    
    // Run first check after 1 minute
    setTimeout(async () => {
        console.log('🔍 Running initial auto-training check...');
        await checkAndTriggerAutoTraining();
    }, 60000);
    
    // Then check every X hours
    setInterval(async () => {
        await checkAndTriggerAutoTraining();
    }, CHECK_INTERVAL_HOURS * 60 * 60 * 1000);
    
    console.log('✅ Auto-training scheduler is ACTIVE!\n');
}

// ============================================================================
// CLEANUP FUNCTIONS
// ============================================================================

async function cleanupDatabaseCache(minutesOld = 60) {
    try {
        console.log(`\n🧹 Database Cleanup (${new Date().toLocaleTimeString()})`);
        console.log(`   Deleting entries older than ${minutesOld} minutes`);
        
        const [before] = await db.query('SELECT COUNT(*) as total FROM predictions');
        console.log(`   Before: ${before[0].total} entries`);
        
        const [result] = await db.query(
            'DELETE FROM predictions WHERE TIMESTAMPDIFF(MINUTE, created_at, NOW()) >= ?',
            [minutesOld]
        );
        
        if (result.affectedRows > 0) {
            console.log(`   ✅ Deleted ${result.affectedRows} entries`);
        } else {
            console.log(`   ℹ️  No old entries found`);
        }
        
        const [after] = await db.query('SELECT COUNT(*) as total FROM predictions');
        console.log(`   After: ${after[0].total} entries remaining\n`);
        
        return result.affectedRows;
        
    } catch (error) {
        console.error('❌ Database cleanup error:', error.message);
        return 0;
    }
}

function cleanupUploadedFiles(hoursOld = 2) {
    const uploadDir = path.join(__dirname, 'uploads');
    const now = Date.now();
    const maxAge = hoursOld * 60 * 60 * 1000;
    
    console.log(`\n🧹 File Cleanup (${new Date().toLocaleTimeString()})`);
    console.log(`   Deleting files older than ${hoursOld} hours`);
    
    try {
        if (!fs.existsSync(uploadDir)) {
            console.log('   ℹ️  Upload directory does not exist\n');
            return 0;
        }
        
        const files = fs.readdirSync(uploadDir);
        
        if (files.length === 0) {
            console.log('   ℹ️  No files\n');
            return 0;
        }
        
        console.log(`   Found: ${files.length} file(s)`);
        
        let deletedCount = 0;
        let keptCount = 0;
        
        files.forEach(file => {
            try {
                const filePath = path.join(uploadDir, file);
                const stats = fs.statSync(filePath);
                const fileAge = now - stats.mtimeMs;
                
                if (fileAge >= maxAge) {
                    fs.unlinkSync(filePath);
                    deletedCount++;
                }  else {
                    keptCount++;
                }
            } catch (err) {}
        });
        
        if (deletedCount > 0) {
            console.log(`   ✅ Deleted: ${deletedCount} file(s)`);
        } else {
            console.log(`   ℹ️  No files old enough`);
        }
        
        if (keptCount > 0) {
            console.log(`   📦 Kept: ${keptCount} recent file(s)`);
        }
        
        console.log('');
        return deletedCount;
        
    } catch (err) {
        console.error('❌ File cleanup error:', err.message);
        return 0;
    }
}

async function performCompleteCleanup() {
    console.log('\n' + '='.repeat(60));
    console.log('🧹 AUTOMATIC CLEANUP');
    console.log('='.repeat(60));
    console.log(`⏰ ${new Date().toLocaleString()}`);
    
    const fileRetentionHours = parseFloat(process.env.FILE_RETENTION_HOURS) || 2;
    const cacheRetentionHours = parseFloat(process.env.CACHE_RETENTION_HOURS) || 24;
    
    const filesDeleted = cleanupUploadedFiles(fileRetentionHours);
    const cacheDeleted = await cleanupDatabaseCache(cacheRetentionHours * 60);
    
    console.log('='.repeat(60));
    console.log(`Summary: ${filesDeleted} files, ${cacheDeleted} cache entries deleted`);
    console.log('='.repeat(60) + '\n');
}

function startCleanupScheduler() {
    const intervalMinutes = parseInt(process.env.CLEANUP_INTERVAL_MINUTES) || 30;
    
    console.log('\n' + '='.repeat(60));
    console.log('🤖 CLEANUP SCHEDULER');
    console.log('='.repeat(60));
    console.log(`⏰ Runs every: ${intervalMinutes} minutes`);
    console.log(`📁 File retention: ${process.env.FILE_RETENTION_HOURS || 2} hours`);
    console.log(`🗄️  Cache retention: ${process.env.CACHE_RETENTION_HOURS || 24} hours`);
    console.log(`🚀 First cleanup: In 5 seconds`);
    console.log('='.repeat(60) + '\n');
    
    setTimeout(async () => {
        console.log('🚀 Running initial cleanup...');
        await performCompleteCleanup();
    }, 5000);
    
    setInterval(async () => {
        await performCompleteCleanup();
    }, intervalMinutes * 60 * 1000);
    
    console.log('✅ Cleanup scheduler active!\n');
}

// ============================================================================
// API ENDPOINTS
// ============================================================================

app.get('/api/health', (req, res) => {
    res.json({ 
        status: 'ok', 
        timestamp: new Date().toISOString(),
        uptime: Math.round(process.uptime())
    });
});

app.post('/api/predict', upload.single('image'), async (req, res) => {
    console.log('\n' + '='.repeat(60));
    console.log('📸 NEW PREDICTION REQUEST');
    console.log('='.repeat(60));
    
    if (!req.file) {
        console.log('❌ No file');
        console.log('='.repeat(60) + '\n');
        return res.status(400).json({ success: false, error: 'No file' });
    }

    const imagePath = req.file.path;
    const imageName = req.file.originalname;
    const startTime = Date.now();

    console.log(`📁 File: ${imageName}`);
    console.log(`📏 Size: ${(req.file.size / 1024).toFixed(2)} KB`);

    try {
        console.log('🔐 Generating hash...');
        const imageHash = await generateImageHash(imagePath);
        console.log(`   Hash: ${imageHash}`);
        
        console.log('🔍 Checking cache...');
        const [cached] = await db.query('SELECT * FROM predictions WHERE image_hash = ?', [imageHash]);
        
        if (cached.length > 0) {
            console.log('✅ CACHE HIT');
            console.log(`   Result: ${cached[0].prediction} (${cached[0].confidence}%)`);
            
            await db.query(
                'UPDATE predictions SET last_accessed = NOW(), access_count = access_count + 1 WHERE image_hash = ?', 
                [imageHash]
            );
            
            console.log('💾 File kept for retention period');
            
            const processingTime = Date.now() - startTime;
            console.log(`⏱️  ${processingTime}ms (from cache)`);
            console.log('='.repeat(60) + '\n');
            
            let probabilities;
            try {
                if (typeof cached[0].probabilities === 'string') {
                    probabilities = JSON.parse(cached[0].probabilities);
                } else if (typeof cached[0].probabilities === 'object') {
                    probabilities = cached[0].probabilities;
                } else {
                    const conf = parseFloat(cached[0].confidence);
                    probabilities = cached[0].prediction === 'FAKE' 
                        ? { FAKE: conf, REAL: 100 - conf }
                        : { REAL: conf, FAKE: 100 - conf };
                }
            } catch (err) {
                const conf = parseFloat(cached[0].confidence);
                probabilities = cached[0].prediction === 'FAKE' 
                    ? { FAKE: conf, REAL: 100 - conf }
                    : { REAL: conf, FAKE: 100 - conf };
            }
            
            return res.json({
                success: true,
                prediction: cached[0].prediction,
                confidence: parseFloat(cached[0].confidence),
                probabilities: probabilities,
                processingTime: processingTime,
                timestamp: cached[0].created_at,
                cached: true
            });
        }

        console.log('❌ CACHE MISS - Running AI...');
        
        const pythonScript = path.join(__dirname, 'python', 'prediction.py');
        
        if (!fs.existsSync(pythonScript)) {
            console.error(`❌ Python script not found: ${pythonScript}`);
            throw new Error('AI script not found');
        }
        
        const python = spawn('python', [pythonScript, imagePath]);
        let dataString = '';
        let errorString = '';

        python.stdout.on('data', (data) => { dataString += data.toString(); });
        python.stderr.on('data', (data) => { 
            errorString += data.toString();
            const line = data.toString().trim();
            if (line && !line.includes('UserWarning')) {
                console.log('[Python]', line);
            }
        });

        python.on('close', async (code) => {
            console.log('💾 File kept in uploads/ (will auto-delete after retention period)');
            
            const processingTime = Date.now() - startTime;

            if (code !== 0) {
                console.error('❌ Python failed');
                console.log('='.repeat(60) + '\n');
                return res.status(500).json({ success: false, error: 'Prediction failed' });
            }

            try {
                const jsonStart = dataString.indexOf('{');
                if (jsonStart === -1) throw new Error('No JSON');
                const result = JSON.parse(dataString.substring(jsonStart));

                if (!result.success) throw new Error(result.error || 'Failed');

                console.log('✅ Prediction successful');
                console.log(`   Result: ${result.prediction} (${result.confidence}%)`);

                // ✅ Auto-add to training data if high confidence
                const autoAdded = await autoAddToTrainingData(
                    imagePath,
                    imageName,
                    result.prediction,
                    result.confidence,
                    imageHash
                );

                try {
                    await db.query(
                        `INSERT INTO predictions 
                        (image_name, image_hash, prediction, confidence, probabilities, processing_time, 
                         created_at, last_accessed, access_count, auto_added_to_training) 
                        VALUES (?, ?, ?, ?, ?, ?, NOW(), NOW(), 1, ?)`,
                        [imageName, imageHash, result.prediction, result.confidence, 
                         JSON.stringify(result.probabilities), processingTime, autoAdded ? 1 : 0]
                    );
                    console.log('💾 Cached');
                } catch (dbError) {
                    console.warn('⚠️  Not cached:', dbError.message);
                }

                console.log(`⏱️  ${processingTime}ms`);
                console.log('='.repeat(60) + '\n');
                
                res.json({
                    success: true,
                    prediction: result.prediction,
                    confidence: result.confidence,
                    probabilities: result.probabilities,
                    processingTime: processingTime,
                    timestamp: new Date().toISOString(),
                    cached: false,
                    autoAddedToTraining: autoAdded
                });
                
            } catch (err) {
                console.error('❌ Parse error:', err.message);
                console.log('='.repeat(60) + '\n');
                res.status(500).json({ success: false, error: 'Parse error' });
            }
        });

    } catch (error) {
        console.error('❌ Error:', error.message);
        console.log('='.repeat(60) + '\n');
        
        try { 
            if (fs.existsSync(imagePath)) fs.unlinkSync(imagePath); 
        } catch (e) {}
        
        res.status(500).json({ success: false, error: error.message });
    }
});

// ============================================================================
// AUTO-TRAINING MONITORING APIs
// ============================================================================

// Get auto-training status
app.get('/api/admin/auto-training-status', async (req, res) => {
    try {
        const trainingDir = path.join(__dirname, 'training_data');
        const fakeDir = path.join(trainingDir, 'FAKE');
        const realDir = path.join(trainingDir, 'REAL');
        
        let fakeCount = 0;
        let realCount = 0;
        
        if (fs.existsSync(fakeDir)) {
            fakeCount = fs.readdirSync(fakeDir).filter(f => /\.(jpg|jpeg|png)$/i.test(f)).length;
        }
        if (fs.existsSync(realDir)) {
            realCount = fs.readdirSync(realDir).filter(f => /\.(jpg|jpeg|png)$/i.test(f)).length;
        }
        
        const totalImages = fakeCount + realCount;
        const balance = totalImages > 0 ? Math.min(fakeCount, realCount) / totalImages : 0;
        
        const [lastTraining] = await db.query(
            'SELECT * FROM training_history ORDER BY trained_at DESC LIMIT 1'
        );
        
        const [autoAddedCount] = await db.query(
            'SELECT COUNT(*) as count FROM training_log WHERE auto_added = 1'
        );
        
        const minImages = parseInt(process.env.MIN_TRAINING_IMAGES) || 50;
        const minBalance = parseFloat(process.env.MIN_BALANCE_RATIO) || 0.3;
        
        const readyForTraining = totalImages >= minImages && 
                                 balance >= minBalance &&
                                 fakeCount >= 10 && 
                                 realCount >= 10;
        
        res.json({
            success: true,
            enabled: process.env.AUTO_TRAINING_ENABLED === 'true',
            currentData: {
                fake: fakeCount,
                real: realCount,
                total: totalImages,
                balance: `${(balance * 100).toFixed(1)}%`
            },
            criteria: {
                minImages: minImages,
                minBalance: `${(minBalance * 100).toFixed(0)}%`,
                minConfidence: `${process.env.HIGH_CONFIDENCE_THRESHOLD || 95}%`,
                checkInterval: `${process.env.TRAINING_CHECK_INTERVAL_HOURS || 6} hours`
            },
            lastTraining: lastTraining[0] || null,
            autoAddedImages: autoAddedCount[0].count,
            readyForTraining: readyForTraining,
            nextCheck: `Within ${process.env.TRAINING_CHECK_INTERVAL_HOURS || 6} hours`
        });
        
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// View auto-added images
app.get('/api/admin/auto-added-images', async (req, res) => {
    try {
        const limit = parseInt(req.query.limit) || 100;
        
        const [images] = await db.query(`
            SELECT 
                tl.*,
                p.image_name as original_name,
                p.created_at as prediction_date
            FROM training_log tl
            LEFT JOIN predictions p ON tl.image_hash = p.image_hash
            WHERE tl.auto_added = 1
            ORDER BY tl.added_at DESC
            LIMIT ?
        `, [limit]);
        
        res.json({
            success: true,
            images: images,
            count: images.length
        });
        
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// View training history
app.get('/api/admin/training-history', async (req, res) => {
    try {
        const [history] = await db.query(`
            SELECT *
            FROM training_history
            ORDER BY trained_at DESC
            LIMIT 20
        `);
        
        res.json({
            success: true,
            history: history,
            count: history.length
        });
        
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Force trigger auto-training check
app.post('/api/admin/trigger-training-check', async (req, res) => {
    try {
        console.log('\n🔍 Manual training check triggered via API...');
        const triggered = await checkAndTriggerAutoTraining();
        
        res.json({
            success: true,
            triggered: triggered,
            message: triggered 
                ? 'Training was triggered!' 
                : 'Not enough data or criteria not met'
        });
        
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Get training statistics
app.get('/api/admin/training-stats', async (req, res) => {
    try {
        const [stats] = await db.query('SELECT * FROM training_stats');
        const [dataStats] = await db.query('SELECT * FROM training_data_summary');
        
        res.json({
            success: true,
            trainingStats: stats[0] || {},
            dataStats: dataStats[0] || {}
        });
        
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Other existing endpoints
app.get('/api/predictions', async (req, res) => {
    try {
        const [rows] = await db.query(`
            SELECT *,
                TIMESTAMPDIFF(MINUTE, created_at, NOW()) as age_minutes
            FROM predictions 
            ORDER BY last_accessed DESC 
            LIMIT 100
        `);
        
        res.json({ success: true, count: rows.length, predictions: rows });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

app.get('/api/admin/cleanup-now', async (req, res) => {
    try {
        console.log('\n🧹 Manual cleanup via API\n');
        
        const filesDeleted = cleanupUploadedFiles(parseFloat(process.env.FILE_RETENTION_HOURS) || 2);
        const cacheDeleted = await cleanupDatabaseCache((parseFloat(process.env.CACHE_RETENTION_HOURS) || 24) * 60);
        
        res.json({
            success: true,
            filesDeleted: filesDeleted,
            cacheDeleted: cacheDeleted,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

app.get('/api/admin/cache-stats', async (req, res) => {
    try {
        const [stats] = await db.query(`
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN prediction = 'FAKE' THEN 1 END) as fake,
                COUNT(CASE WHEN prediction = 'REAL' THEN 1 END) as real,
                AVG(confidence) as avg_confidence
            FROM predictions
        `);
        
        res.json({ success: true, stats: stats[0] });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, '..', 'frontend', 'index.html'));
});

// ============================================================================
// START SERVER
// ============================================================================

const PORT = parseInt(process.env.PORT) || 3000;

app.listen(PORT, () => {
    console.log('\n' + '='.repeat(60));
    console.log('🚀 DEEPFAKE DETECTION SERVER WITH AUTO-TRAINING');
    console.log('='.repeat(60));
    console.log(`📡 Server:  http://localhost:${PORT}`);
    console.log(`🔍 Predict: http://localhost:${PORT}/api/predict`);
    console.log(`📊 Status:  http://localhost:${PORT}/api/admin/auto-training-status`);
    console.log(`📜 History: http://localhost:${PORT}/api/admin/training-history`);
    console.log('='.repeat(60) + '\n');
    
    setTimeout(() => {
        startCleanupScheduler();
        startAutoTrainingScheduler();
    }, 2000);
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n\n👋 Shutting down gracefully...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n\n👋 Shutting down gracefully...');
    process.exit(0);
});