const mysql = require('mysql2');
require('dotenv').config();

// Log configuration (for debugging)
console.log('📊 Database Configuration:');
console.log('   Host:', process.env.DB_HOST || 'localhost');
console.log('   User:', process.env.DB_USER || 'root');
console.log('   Database:', process.env.DB_NAME || 'deepfake_detection');
console.log('   Port:', process.env.DB_PORT || 3306);

const pool = mysql.createPool({
    host: process.env.DB_HOST || 'localhost',
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || 'root',
    database: process.env.DB_NAME || 'deepfake_detection', // Make sure this is correct
    port: process.env.DB_PORT || 3306,
    waitForConnections: true,
    connectionLimit: 10,
    queueLimit: 0
});

// Test connection
pool.getConnection((err, connection) => {
    if (err) {
        console.error('✗ MySQL connection error:', err.message);
        console.error('   Error code:', err.code);
        if (err.code === 'ER_BAD_DB_ERROR') {
            console.error('   → Database does not exist. Please create it first.');
            console.error('   → Run: CREATE DATABASE deepfake_detection;');
        }
    } else {
        console.log('✅ MySQL Connected Successfully');
        console.log('   Connection ID:', connection.threadId);
        connection.release();
    }
});

module.exports = pool.promise();