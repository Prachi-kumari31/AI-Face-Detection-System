require('dotenv').config();
const fs = require('fs');
const path = require('path');
const db = require('./database');

async function testSetup() {
    console.log('\n🧪 TESTING AUTO-TRAINING SETUP\n');
    console.log('='.repeat(60));
    
    let allGood = true;
    
    // Test 1: Python script
    console.log('\n1️⃣ Checking Python training script...');
    const pythonScript = path.join(__dirname, 'python', 'train_model.py');
    
    if (fs.existsSync(pythonScript)) {
        console.log('   ✅ train_model.py exists');
    } else {
        console.log('   ❌ train_model.py NOT FOUND');
        console.log(`   Expected: ${pythonScript}`);
        allGood = false;
    }
    
    // Test 2: Training folders
    console.log('\n2️⃣ Checking training folders...');
    const trainingDir = path.join(__dirname, 'training_data');
    const fakeDir = path.join(trainingDir, 'FAKE');
    const realDir = path.join(trainingDir, 'REAL');
    
    if (fs.existsSync(fakeDir)) {
        const fakeCount = fs.readdirSync(fakeDir).length;
        console.log(`   ✅ training_data/FAKE/ exists (${fakeCount} files)`);
    } else {
        console.log('   ❌ training_data/FAKE/ NOT FOUND');
        allGood = false;
    }
    
    if (fs.existsSync(realDir)) {
        const realCount = fs.readdirSync(realDir).length;
        console.log(`   ✅ training_data/REAL/ exists (${realCount} files)`);
    } else {
        console.log('   ❌ training_data/REAL/ NOT FOUND');
        allGood = false;
    }
    
    // Test 3: Model file
    console.log('\n3️⃣ Checking model file...');
    const modelPath = path.join(__dirname, '..', 'models', 'deepfake_detector_cnn.pth');
    
    if (fs.existsSync(modelPath)) {
        const stats = fs.statSync(modelPath);
        console.log(`   ✅ Model exists (${(stats.size / (1024*1024)).toFixed(2)} MB)`);
    } else {
        console.log('   ⚠️  Model file not found (will be created during first training)');
    }
    
    // Test 4: Database columns
    console.log('\n4️⃣ Checking database schema...');
    
    try {
        const [columns] = await db.query(`
            SELECT COLUMN_NAME 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = 'deepfake_detection' 
            AND TABLE_NAME = 'training_history'
            AND COLUMN_NAME IN ('accuracy', 'files_deleted')
        `);
        
        const hasAccuracy = columns.some(c => c.COLUMN_NAME === 'accuracy');
        const hasFilesDeleted = columns.some(c => c.COLUMN_NAME === 'files_deleted');
        
        if (hasAccuracy) {
            console.log('   ✅ training_history.accuracy column exists');
        } else {
            console.log('   ❌ training_history.accuracy column MISSING');
            console.log('      Run: ALTER TABLE training_history ADD COLUMN accuracy DECIMAL(5,2) NULL;');
            allGood = false;
        }
        
        if (hasFilesDeleted) {
            console.log('   ✅ training_history.files_deleted column exists');
        } else {
            console.log('   ❌ training_history.files_deleted column MISSING');
            console.log('      Run: ALTER TABLE training_history ADD COLUMN files_deleted INT DEFAULT 0;');
            allGood = false;
        }
        
    } catch (error) {
        console.log('   ❌ Database check failed:', error.message);
        allGood = false;
    }
    
    // Test 5: Check server.js has function
    console.log('\n5️⃣ Checking server.js code...');
    const serverPath = path.join(__dirname, 'server.js');
    const serverCode = fs.readFileSync(serverPath, 'utf8');
    
    if (serverCode.includes('deleteTrainingFilesAfterTraining')) {
        console.log('   ✅ deleteTrainingFilesAfterTraining function exists');
    } else {
        console.log('   ❌ deleteTrainingFilesAfterTraining function MISSING');
        console.log('      Add the function to server.js');
        allGood = false;
    }
    
    if (serverCode.includes('train_model.py')) {
        console.log('   ✅ triggerAutomaticTraining calls Python script');
    } else {
        console.log('   ⚠️  triggerAutomaticTraining might not call train_model.py');
    }
    
    // Summary
    console.log('\n' + '='.repeat(60));
    
    if (allGood) {
        console.log('✅ ALL CHECKS PASSED!');
        console.log('   System is ready for auto-training');
    } else {
        console.log('❌ SOME CHECKS FAILED');
        console.log('   Fix the issues above before running');
    }
    
    console.log('='.repeat(60) + '\n');
    
    process.exit(0);
}

testSetup();