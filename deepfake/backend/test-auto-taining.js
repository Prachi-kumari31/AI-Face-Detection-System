require('dotenv').config();
const fs = require('fs');
const path = require('path');
const db = require('./database');

async function testAutoTraining() {
    console.log('\n🧪 TESTING AUTO-TRAINING SETUP\n');
    console.log('='.repeat(60));
    
    try {
        // Test 1: Check environment variables
        console.log('\n1️⃣ Environment Variables:');
        console.log(`   AUTO_TRAINING_ENABLED: ${process.env.AUTO_TRAINING_ENABLED}`);
        console.log(`   HIGH_CONFIDENCE_THRESHOLD: ${process.env.HIGH_CONFIDENCE_THRESHOLD}`);
        console.log(`   MIN_TRAINING_IMAGES: ${process.env.MIN_TRAINING_IMAGES}`);
        console.log(`   MIN_BALANCE_RATIO: ${process.env.MIN_BALANCE_RATIO}`);
        console.log(`   TRAINING_CHECK_INTERVAL_HOURS: ${process.env.TRAINING_CHECK_INTERVAL_HOURS}`);
        
        // Test 2: Check folder structure
        console.log('\n2️⃣ Folder Structure:');
        const trainingDir = path.join(__dirname, 'training_data');
        const fakeDir = path.join(trainingDir, 'FAKE');
        const realDir = path.join(trainingDir, 'REAL');
        
        console.log(`   training_data/: ${fs.existsSync(trainingDir) ? '✅' : '❌'}`);
        console.log(`   training_data/FAKE/: ${fs.existsSync(fakeDir) ? '✅' : '❌'}`);
        console.log(`   training_data/REAL/: ${fs.existsSync(realDir) ? '✅' : '❌'}`);
        
        if (fs.existsSync(fakeDir)) {
            const fakeCount = fs.readdirSync(fakeDir).length;
            console.log(`      FAKE images: ${fakeCount}`);
        }
        if (fs.existsSync(realDir)) {
            const realCount = fs.readdirSync(realDir).length;
            console.log(`      REAL images: ${realCount}`);
        }
        
        // Test 3: Check database tables
        console.log('\n3️⃣ Database Tables:');
        
        try {
            const [trainingLog] = await db.query('SELECT COUNT(*) as count FROM training_log');
            console.log(`   training_log: ✅ (${trainingLog[0].count} entries)`);
        } catch (e) {
            console.log(`   training_log: ❌ ${e.message}`);
        }
        
        try {
            const [trainingHistory] = await db.query('SELECT COUNT(*) as count FROM training_history');
            console.log(`   training_history: ✅ (${trainingHistory[0].count} entries)`);
        } catch (e) {
            console.log(`   training_history: ❌ ${e.message}`);
        }
        
        try {
            const [predictions] = await db.query('SHOW COLUMNS FROM predictions LIKE "auto_added_to_training"');
            console.log(`   predictions.auto_added_to_training: ${predictions.length > 0 ? '✅' : '❌'}`);
        } catch (e) {
            console.log(`   predictions column: ❌ ${e.message}`);
        }
        
        // Test 4: Summary
        console.log('\n4️⃣ System Status:');
        const allGood = 
            process.env.AUTO_TRAINING_ENABLED === 'true' &&
            fs.existsSync(fakeDir) &&
            fs.existsSync(realDir);
        
        if (allGood) {
            console.log('   ✅ Auto-training system is ready!');
        } else {
            console.log('   ⚠️  Some components need setup');
        }
        
        console.log('\n' + '='.repeat(60));
        console.log('✅ TEST COMPLETE\n');
        
        process.exit(0);
        
    } catch (error) {
        console.error('\n❌ Test error:', error.message);
        process.exit(1);
    }
}

testAutoTraining();