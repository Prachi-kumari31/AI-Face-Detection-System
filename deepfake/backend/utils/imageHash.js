const crypto = require('crypto');
const fs = require('fs');

/**
 * Generate MD5 hash of image file for caching
 */
function generateImageHash(filePath) {
    return new Promise((resolve, reject) => {
        const hash = crypto.createHash('md5');
        const stream = fs.createReadStream(filePath);
        
        stream.on('data', (data) => {
            hash.update(data);
        });
        
        stream.on('end', () => {
            resolve(hash.digest('hex'));
        });
        
        stream.on('error', (error) => {
            reject(error);
        });
    });
}

module.exports = { generateImageHash };
